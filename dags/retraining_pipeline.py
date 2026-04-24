from datetime import datetime
import json
import os
from urllib import error, request

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from mlops_pipeline.drift import detect_drift
from mlops_pipeline.pipeline import compare_and_promote_model, train_and_register_model


DEFAULT_REFERENCE_DATA = os.getenv("REFERENCE_DATA_PATH", "/opt/airflow/project/data/medium_articles.csv")
DEFAULT_CURRENT_DATA = os.getenv("CURRENT_DATA_PATH", "/opt/airflow/project/data/medium_articles.csv")
DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "/opt/airflow/project/distilbert_tagging_model")
DEFAULT_REPORT_DIR = os.getenv("DRIFT_REPORT_DIR", "/opt/airflow/project/reports")
DEFAULT_REFERENCE_PREDICTIONS = os.getenv("REFERENCE_PREDICTIONS_PATH")
DEFAULT_CURRENT_PREDICTIONS = os.getenv("CURRENT_PREDICTIONS_PATH")
DEFAULT_OOV_HISTORY = os.getenv("OOV_HISTORY_PATH")
DEFAULT_CTR_HISTORY = os.getenv("CTR_HISTORY_PATH")
DEFAULT_CURRENT_FEEDBACK = os.getenv("CURRENT_FEEDBACK_PATH")
DEFAULT_MODEL_RELOAD_URL = os.getenv("MODEL_RELOAD_URL", "http://api:8000/admin/reload-model")
DEFAULT_MODEL_RELOAD_TOKEN = os.getenv("MODEL_RELOAD_TOKEN")


def _dag_conf(defaults):
    context = get_current_context()
    dag_run = context.get("dag_run")
    conf = {} if dag_run is None or dag_run.conf is None else dict(dag_run.conf)
    return {key: conf.get(key, value) for key, value in defaults.items()}


@dag(
    dag_id="article_tagger_retraining",
    schedule="0 0 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "drift", "retraining"],
)
def article_tagger_retraining():
    @task.short_circuit
    def drift_gate(
        reference_data_path=DEFAULT_REFERENCE_DATA,
        current_data_path=DEFAULT_CURRENT_DATA,
        label_frequency_threshold=0.15,
        oov_sigma_threshold=2.0,
        ctr_drop_threshold=0.20,
    ):
        config = _dag_conf(
            {
                "reference_data_path": reference_data_path,
                "current_data_path": current_data_path,
                "reference_prediction_path": DEFAULT_REFERENCE_PREDICTIONS,
                "current_prediction_path": DEFAULT_CURRENT_PREDICTIONS,
                "oov_history_path": DEFAULT_OOV_HISTORY,
                "ctr_history_path": DEFAULT_CTR_HISTORY,
                "current_feedback_path": DEFAULT_CURRENT_FEEDBACK,
                "label_frequency_threshold": label_frequency_threshold,
                "oov_sigma_threshold": oov_sigma_threshold,
                "ctr_drop_threshold": ctr_drop_threshold,
                "top_k_tags": 50,
                "drift_report_dir": DEFAULT_REPORT_DIR,
            }
        )
        report_path = os.path.join(config["drift_report_dir"], "drift_report.json")
        report = detect_drift(
            reference_path=config["reference_data_path"],
            current_path=config["current_data_path"],
            report_path=report_path,
            reference_prediction_path=config["reference_prediction_path"],
            current_prediction_path=config["current_prediction_path"],
            oov_history_path=config["oov_history_path"],
            ctr_history_path=config["ctr_history_path"],
            current_feedback_path=config["current_feedback_path"],
            label_frequency_threshold=config["label_frequency_threshold"],
            oov_sigma_threshold=config["oov_sigma_threshold"],
            ctr_drop_threshold=config["ctr_drop_threshold"],
            top_k_tags=config["top_k_tags"],
        )
        return report["drift_detected"]

    @task
    def retrain_model(
        current_data_path=DEFAULT_CURRENT_DATA,
        epochs=2,
        batch_size=4,
        lr=5e-5,
        model_dir=DEFAULT_MODEL_DIR,
    ):
        config = _dag_conf(
            {
                "current_data_path": current_data_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "model_dir": model_dir,
            }
        )
        return train_and_register_model(
            csv_path=config["current_data_path"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            model_dir=config["model_dir"],
            run_name="Airflow_Retrain",
        )

    @task
    def evaluate_candidate(training_summary):
        return {
            "run_id": training_summary["run_id"],
            "model_version": training_summary["model_version"],
            "metrics": training_summary["metrics"],
            "model_uri": training_summary["model_uri"],
        }

    @task
    def compare_and_promote(candidate_summary, metric_name="micro_f1"):
        config = _dag_conf({"metric_name": metric_name})
        return compare_and_promote_model(
            candidate_version=candidate_summary["model_version"],
            metric_name=config["metric_name"],
        )

    @task
    def reload_serving_model(
        promotion_summary,
        model_reload_url=DEFAULT_MODEL_RELOAD_URL,
        model_reload_token=DEFAULT_MODEL_RELOAD_TOKEN,
    ):
        if not promotion_summary["promoted"]:
            return {"reloaded": False, "reason": "candidate_not_promoted"}
        config = _dag_conf(
            {
                "model_reload_url": model_reload_url,
                "model_reload_token": model_reload_token,
            }
        )
        headers = {}
        if config["model_reload_token"]:
            headers["X-Model-Reload-Token"] = config["model_reload_token"]
        headers["Content-Type"] = "application/json"
        payload = json.dumps({"expected_model_version": promotion_summary["candidate_version"]}).encode("utf-8")
        req = request.Request(config["model_reload_url"], data=payload, method="POST", headers=headers)
        try:
            with request.urlopen(req, timeout=30) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Model reload request failed url={config['model_reload_url']} status={exc.code} body={body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Model reload request failed url={config['model_reload_url']} reason={exc.reason}"
            ) from exc
        return {
            "reloaded": True,
            "status_code": response.status,
            "response": body,
        }

    gate = drift_gate()
    training = retrain_model()
    evaluation = evaluate_candidate(training)
    promotion = compare_and_promote(evaluation)
    reload_model = reload_serving_model(promotion)

    gate >> training
    training >> evaluation >> promotion >> reload_model


article_tagger_retraining()
