import sys
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()


# ============================================================
# Base Directory
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# Project Imports
# ============================================================

from ml_model import predict_rul

from tools import (
    query_engine_cost_sql,
    calculate_rul_and_maintenance_cost_api,
    send_user_alert_api,
)


# ============================================================
# DeepSeek Model
# ============================================================

deepseek_model = ChatOpenAI(
    model="deepseek-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ============================================================
# LangChain Tools
# ============================================================


@tool
def engine_prediction_tool(
    engine_id: str,
    cycle: int,
    sensor_1: float,
    sensor_2: float,
    sensor_3: float,
    sensor_4: float,
    sensor_5: float,
    sensor_6: float,
    sensor_7: float,
    sensor_8: float,
    sensor_9: float,
    sensor_10: float,
    sensor_11: float,
    sensor_12: float,
    sensor_13: float,
    sensor_14: float,
) -> dict:
    """
    Predict the Remaining Useful Life (RUL) of a turbofan engine
    using the trained Bagging regression model.
    """

    sensor_values = [
        sensor_1,
        sensor_2,
        sensor_3,
        sensor_4,
        sensor_5,
        sensor_6,
        sensor_7,
        sensor_8,
        sensor_9,
        sensor_10,
        sensor_11,
        sensor_12,
        sensor_13,
        sensor_14,
    ]

    predicted_rul = predict_rul(
        engine_id=engine_id,
        sensor_values=sensor_values,
        cycle=cycle,
    )

    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "predicted_rul_cycles": round(
            predicted_rul,
            2
        ),
        "model": "Bagging Regressor",
        "task": "RUL Regression",
    }


@tool
def fleet_registry_tool(
    engine_id: str
) -> dict:
    """
    Retrieve engine asset information from the fleet registry.
    """

    return query_engine_cost_sql(
        engine_id
    )


@tool
def rul_degradation_tool(
    engine_id: str,
    predicted_rul: float
) -> dict:
    """
    Determine health status, risk level,
    and estimated repair cost using the ML-predicted RUL.
    """

    return calculate_rul_and_maintenance_cost_api(
        engine_id,
        predicted_rul
    )


@tool
def notification_tool(
    engine_id: str,
    health_status: str,
    asset_cost: float,
    repair_cost: float,
    predicted_rul: float,
    risk_level: str,
) -> dict:
    """
    Create the final predictive maintenance notification.
    """

    return send_user_alert_api(
        engine_id=engine_id,
        health_status=health_status,
        asset_cost=asset_cost,
        repair_cost=repair_cost,
        predicted_rul=predicted_rul,
        risk_level=risk_level,
    )


# ============================================================
# Supervisor System Prompt
# ============================================================

SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Agent for the Raseen predictive maintenance system.

Your responsibility is to coordinate the complete predictive maintenance
workflow for a NASA C-MAPSS turbofan engine.

Follow these steps in order.

STEP 1:
Use the Engine Prediction Tool.

The tool receives:
- Engine ID
- Cycle
- Sensor 1
- Sensor 2
- Sensor 3
- Sensor 4
- Sensor 5
- Sensor 6
- Sensor 7
- Sensor 8
- Sensor 9
- Sensor 10
- Sensor 11
- Sensor 12
- Sensor 13
- Sensor 14

The tool uses the trained Bagging Regressor to predict
the Remaining Useful Life (RUL).

STEP 2:
Use the Fleet Registry Tool with the engine ID.

Retrieve the engine asset ID and asset cost.

STEP 3:
Use the RUL Degradation Tool.

Pass:
- Engine ID
- The predicted RUL returned by the Engine Prediction Tool

The tool determines:
- Health Status
- Risk Level
- Estimated Repair Cost


STEP 4:
Use the Notification Tool to generate a local maintenance alert only.

The notification is displayed or returned within Raseen.
Do not claim that an external notification was sent.
Do not send commands to industrial equipment.
Do not initiate outbound connections to equipment or OT networks.

STEP 5:
Do not skip any required tool.

STEP 6:
Do not invent sensor values, RUL values, asset costs,
repair costs, health statuses, risk levels, or notification results.

STEP 7:
Use the output of each tool as input for the next tool.

STEP 8:
After all tools have been completed, provide a concise final report
containing:

- Engine ID
- Cycle
- Health Status
- Model Used
- Predicted RUL
- Risk Level
- Asset Cost
- Repair Cost
- Final Maintenance Message

STEP 9:
Raseen operates in receive-only analysis mode.
Process the provided sensor data and generate local analysis results.
Never issue equipment-control commands or claim that physical
one-way communication is enforced by software alone.
Report the notification status as LOCAL_ONLY.

The RUL must come from the Machine Learning model.
Do not calculate or invent another RUL value.
"""




# ============================================================
# Build Supervisor
# ============================================================

def build_supervisor(model):
    """
    Build the LangChain Supervisor Agent.

    The model is passed from outside so the LLM
    can be changed without changing the workflow.
    """

    supervisor = create_agent(
        model=model,
        tools=[
            engine_prediction_tool,
            fleet_registry_tool,
            rul_degradation_tool,
            notification_tool,
        ],
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        name="raseen_supervisor",
    )

    return supervisor


# ============================================================
# Run Supervisor
# ============================================================

def run_supervisor(
    engine_id: str,
    cycle: int,
    sensor_1: float,
    sensor_2: float,
    sensor_3: float,
    sensor_4: float,
    sensor_5: float,
    sensor_6: float,
    sensor_7: float,
    sensor_8: float,
    sensor_9: float,
    sensor_10: float,
    sensor_11: float,
    sensor_12: float,
    sensor_13: float,
    sensor_14: float,
):
    """
    Run the complete Raseen predictive maintenance workflow.
    """

    logs = []

    # --------------------------------------------------------
    # Input Summary
    # --------------------------------------------------------

    logs.append(
        f"**[Raseen ML Input]:** "
        f"Engine `{engine_id}` | "
        f"Cycle `{cycle}` | "
        f"14 sensor readings received."
    )

    # --------------------------------------------------------
    # Build Supervisor
    # --------------------------------------------------------

    supervisor = build_supervisor(
        deepseek_model
    )

    # --------------------------------------------------------
    # User Request
    # --------------------------------------------------------

    user_message = f"""
Analyze turbofan engine {engine_id}
using the provided NASA C-MAPSS sensor readings.

Engine ID:
{engine_id}

Current Cycle:
{cycle}

Sensor 1:
{sensor_1}

Sensor 2:
{sensor_2}

Sensor 3:
{sensor_3}

Sensor 4:
{sensor_4}

Sensor 5:
{sensor_5}

Sensor 6:
{sensor_6}

Sensor 7:
{sensor_7}

Sensor 8:
{sensor_8}

Sensor 9:
{sensor_9}

Sensor 10:
{sensor_10}

Sensor 11:
{sensor_11}

Sensor 12:
{sensor_12}

Sensor 13:
{sensor_13}

Sensor 14:
{sensor_14}

Complete the full predictive maintenance workflow.

First:
Use the Engine Prediction Tool to predict the Remaining Useful Life (RUL).

Second:
Use the Fleet Registry Tool to retrieve the engine asset information.

Third:
Use the RUL Degradation Tool using the predicted RUL
returned by the Machine Learning model.

Fourth:
Use the Notification Tool to generate the final maintenance alert.

Use the output of each tool as input for the next tool.

Do not invent any values.

The final report must contain:

- Engine ID
- Cycle
- Health Status
- Model Used
- Predicted RUL
- Risk Level
- Asset Cost
- Repair Cost
- Final Maintenance Message
"""

    # --------------------------------------------------------
    # Run Agent
    # --------------------------------------------------------

    agent_result = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        }
    )

    # --------------------------------------------------------
    # Final Agent Response
    # --------------------------------------------------------

    agent_response = (
        agent_result["messages"][-1].content
    )

    logs.append(
        f"**[LangChain Supervisor]:**\n"
        f"{agent_response}"
    )

    return {
        "logs": logs
    }