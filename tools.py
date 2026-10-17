# NASA C-MAPSS Turbofan Engine Fleet & Degradation Metrics

NASA_CMAPSS_FLEET = {
    "ENGINE-001": {
        "engine_id": "FD001_ENG_04",
        "unit_cost": 12000000
    },
    "ENGINE-002": {
        "engine_id": "FD001_ENG_08",
        "unit_cost": 12000000
    }
}


def query_engine_cost_sql(engine_id: str):
    """
    Agent 1 Tool:
    Retrieve NASA C-MAPSS turbofan asset information.
    """

    engine_data = NASA_CMAPSS_FLEET.get(
        engine_id.upper()
    )

    if engine_data is None:
        return {
            "status": "ERROR",
            "message": f"Engine {engine_id} was not found in the fleet registry."
        }

    return {
        "engine_name": engine_id,
        "engine_id": engine_data["engine_id"],
        "asset_cost_usd": engine_data["unit_cost"],
        "source": "NASA C-MAPSS Simulator Fleet Registry"
    }


def calculate_rul_and_maintenance_cost_api(
    engine_id: str,
    predicted_rul: float
):
    """
    Agent 2 Tool:
    Use the ML-predicted RUL to determine health status,
    risk level, and estimated repair cost.
    """

    engine_data = NASA_CMAPSS_FLEET.get(
        engine_id.upper()
    )

    if engine_data is None:
        return {
            "status": "ERROR",
            "message": f"Engine {engine_id} was not found in the fleet registry."
        }

    predicted_rul = max(
        0,
        float(predicted_rul)
    )

    # Health status based on predicted RUL
    if predicted_rul <= 25:

        health_status = "CRITICAL_DEGRADATION"

        risk_level = (
            "Critical Degradation "
            "(Low RUL / Imminent Failure)"
        )

        damage_multiplier = 3.0

    elif predicted_rul <= 50:

        health_status = "WARNING_DEGRADATION"

        risk_level = (
            "Moderate Subsystem Degradation"
        )

        damage_multiplier = 1.4

    else:

        health_status = "NORMAL"

        risk_level = (
            "Healthy / Optimal RUL State"
        )

        damage_multiplier = 0.1

    estimated_overhaul = (
        450000 * damage_multiplier
    )

    return {
        "health_status": health_status,
        "predicted_rul_cycles": round(
            predicted_rul,
            2
        ),
        "risk_level": risk_level,
        "repair_cost_usd": estimated_overhaul
    }



def send_user_alert_api(
    engine_id: str,
    health_status: str,
    asset_cost: float,
    repair_cost: float,
    predicted_rul: float,
    risk_level: str
):
    """
    Agent 3 Tool:
    Generate a local maintenance alert only.
    No external notification or equipment command is sent.
    """

    alert_msg = (
        f"[Raseen Predictive Maintenance Alert] "
        f"Turbofan Engine {engine_id} | "
        f"Health Status: {health_status} | "
        f"Predicted RUL: {predicted_rul:.2f} cycles | "
        f"Risk Level: {risk_level} | "
        f"Engine Asset Value: ${asset_cost:,.2f} | "
        f"Estimated Repair Cost: ${repair_cost:,.2f}"
    )

    return {
        "status": "LOCAL_ONLY",
        "message": alert_msg,
        "external_delivery": False,
        "equipment_command_sent": False
    }
