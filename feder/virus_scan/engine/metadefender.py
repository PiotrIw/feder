import logging
import time

import requests
from django.conf import settings

from feder.virus_scan.models import Request

from .base import BaseEngine

logger = logging.getLogger(__name__)


class MetaDefenderEngine(BaseEngine):
    name = "MetaDefender"

    def __init__(self):
        self.key = settings.METADEFENDER_API_KEY
        self.url = settings.METADEFENDER_API_URL
        self.session = requests.Session()
        super().__init__()

    def map_status(self, resp):
        status = resp.get("status")
        scan_results = resp.get("scan_results", {})
        process_info = resp.get("process_info", {})

        if status == "inqueue":
            return Request.STATUS.queued

        if scan_results.get("scan_all_result_a") == "In queue":
            return Request.STATUS.queued

        if process_info.get("progress_percentage") not in (None, 100):
            return Request.STATUS.queued

        scan_result_i = scan_results.get("scan_all_result_i")
        scan_result_a = scan_results.get("scan_all_result_a")

        if scan_result_i == 0:
            return Request.STATUS.not_detected

        if scan_result_a == "Aborted":
            return Request.STATUS.failed

        if scan_result_i and scan_result_i > 0:
            return Request.STATUS.infected

        return Request.STATUS.failed

    def get_result_url(self, engine_id):
        return f"{self.url}/v4/file/{engine_id}"

    def send_scan(self, this_file, filename):
        try:
            resp = self.session.post(
                f"{self.url}/v4/file",
                files={"": (filename, this_file, "application/octet-stream")},
                headers={
                    "apikey": self.key,
                    "filename": filename.encode("ascii", "ignore"),
                    "callbackurl": self.get_webhook_url(),
                },
            )
            result = resp.json()
            result["response_headers"] = dict(resp.headers)
            resp.raise_for_status()
            return {
                "engine_id": result["data_id"],
                "status": self.map_status(result),
                "engine_report": result,
                "engine_link": self.get_result_url(
                    result["data_id"] if result["data_id"] is not None else None,
                ),
            }

        except requests.exceptions.HTTPError as e:
            result["error"] = str(e)
            if resp.status_code == 429:
                logger.warning(f"Rate limit hit for {filename}: {e}", exc_info=False)
            else:
                logger.error(f"HTTP error for {filename}: {e}", exc_info=False)

            return {
                "status": Request.STATUS.failed,
                "engine_report": result,
            }

        except requests.exceptions.RequestException as e:
            result = result if isinstance(result, dict) else {}
            result["error"] = str(e)
            logger.error(
                f"Failed to send request {filename}: {e}"
                + " - waiting 30 sec before sending next"
            )
            time.sleep(30)
            return {
                "status": Request.STATUS.failed,
                "engine_report": result,
            }

    def receive_result(self, engine_id):
        try:
            resp = self.session.get(
                self.get_result_url(engine_id),
                headers={"apikey": self.key},
            )
            resp.raise_for_status()
            result = dict(resp.json())
            result["response_headers"] = dict(resp.headers)
            return {
                "engine_id": result["data_id"],
                "status": self.map_status(result),
                "engine_report": result,
                "engine_link": self.get_result_url(
                    result["data_id"] if result["data_id"] is not None else None,
                ),
            }
        except requests.exceptions.RequestException as e:
            result["error"] = str(e)
            logger.error(f"Failed to receive result {engine_id}: {e}")
            return {
                "status": Request.STATUS.failed,
                "engine_report": result,
            }
