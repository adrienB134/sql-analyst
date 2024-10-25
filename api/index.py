from fastapi import FastAPI, Request
from .prompt import system_prompt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlite3
import json
import os

device = "cuda"
model_path = "ibm-granite/granite-3.0-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map=device, torch_dtype=torch.bfloat16
)
model.eval()

sql = """create table FINANCE (
	company_name VARCHAR(50),
	revenue DECIMAL(9,2),
	expenses DECIMAL(8,2),
	profit DECIMAL(10,2),
	taxes DECIMAL(8,2),
	net_income DECIMAL(10,2),
	assets DECIMAL(9,2),
	liabilities DECIMAL(9,2),
	equity DECIMAL(10,2),
	date_reported DATE
);"""

if not os.path.exists("./api/db.sqlite"):
    conn = sqlite3.connect("./api/db.sqlite")
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    with open("./api/FINANCE.sql", "r") as f:
        sql = f.readlines()
        for line in sql:
            cursor.execute(line)
    conn.commit()
    cursor.execute("SELECT * FROM FINANCE")
    print(cursor.fetchall())
    conn.close()

tool = {
    "name": "generate_graph_data",
    "description": "Generate structured JSON data for creating financial charts and graphs.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chartType": {
                "type": "string",
                "enum": ["bar", "multiBar", "line", "pie", "area", "stackedArea"],
                "description": "The type of chart to generate",
            },
            "config": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "trend": {
                        "type": "object",
                        "properties": {
                            "percentage": {"type": "number"},
                            "direction": {"type": "string", "enum": ["up", "down"]},
                        },
                        "required": ["percentage", "direction"],
                    },
                    "footer": {"type": "string"},
                    "totalLabel": {"type": "string"},
                    "xAxisKey": {"type": "string"},
                },
                "required": ["title", "description"],
            },
            "data": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "chartConfig": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "stacked": {"type": "boolean"},
                    },
                    "required": ["label"],
                },
            },
        },
        "required": ["chartType", "config", "data", "chartConfig"],
    },
}


def generate_sql(prompt, sql=sql):
    chat = [
        {
            "role": "user",
            "content": """Here is a sql table:
            {table}
            
            Generate a sql query that answers the following question: 
            {prompt}
            
            Output only the query, without any other text.""".format(
                table=sql, prompt=prompt
            ),
        },
    ]
    chat = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    # tokenize the text
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    # generate output tokens
    output = model.generate(**input_tokens, max_new_tokens=2000)
    answer_tokens = output[:, len(input_tokens["input_ids"][0]) :]
    # decode output tokens into text
    output = tokenizer.decode(answer_tokens[0], skip_special_tokens=True)
    print(f"SQL: {output}")
    conn = sqlite3.connect("./api/db.sqlite")
    cursor = conn.cursor()
    cursor.execute(output)
    res = cursor.fetchall()
    print(res)
    conn.close()
    return res, output


def generate_graph_data(res, question, sql_query):
    chat = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": """Here is a sql query:
        {query}
        
        And here are the data from the previous sql query: 
        {results}
        
        And here is a question about that data:
        {question}
        
        Generate a chart from that data that answers the question.
        
        Output only the json, without any other text. Base yourself only on the data provided """.format(
                query=sql_query, results=res, question=question
            ),
        },
    ]
    chat = tokenizer.apply_chat_template(
        chat, tools=[tool], tokenize=False, add_generation_prompt=True
    )
    # tokenize the text
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    # generate output tokens
    output = model.generate(**input_tokens, max_new_tokens=2000)
    answer_tokens = output[:, len(input_tokens["input_ids"][0]) :]
    # decode output tokens into text
    output = tokenizer.decode(answer_tokens[0], skip_special_tokens=True)

    return output


app = FastAPI()


@app.post("/api/py/get_answer")
async def get_answer(request: Request):
    data = await request.json()
    financial_data, sql_query = generate_sql(data["messages"][-1]["content"])
    output = generate_graph_data(
        financial_data, data["messages"][-1]["content"], sql_query
    )
    try:
        processed_output = process_tool_response(json.loads(output))
        print(processed_output)
    except Exception as e:
        print(f"Coloring failed: {e}")
        processed_output = json.loads(output)
        print(processed_output)
    return {
        "content": "Here is the chart",
        "hasToolUse": "true",
        "toolUse": processed_output,
        "chartData": processed_output,
    }


def process_tool_response(chart_data):
    if (
        not chart_data.get("chartType")
        or not chart_data.get("data")
        or not isinstance(chart_data["data"], list)
    ):
        raise ValueError("Invalid chart data structure")

    # Transform data for pie charts to match expected structure
    if chart_data["chartType"] == "pie":
        # Ensure data items have 'segment' and 'value' keys
        chart_data["data"] = [
            {
                "segment": item.get(chart_data["config"].get("xAxisKey", "segment"))
                or item.get("segment")
                or item.get("category")
                or item.get("name"),
                "value": item.get(list(chart_data["chartConfig"].keys())[0])
                or item.get("value"),
            }
            for item in chart_data["data"]
        ]

        # Ensure xAxisKey is set to 'segment' for consistency
        chart_data["config"]["xAxisKey"] = "segment"

    # Create new chartConfig with system color variables
    processed_chart_config = {
        key: {**config, "color": f"hsl(var(--chart-{index + 1}))"}
        for index, (key, config) in enumerate(chart_data["chartConfig"].items())
    }

    return {**chart_data, "chartConfig": processed_chart_config}
