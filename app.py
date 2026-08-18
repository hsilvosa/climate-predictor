"""Root launcher for FastAPI Web App and Hugging Face Spaces."""

import os
import uvicorn
from climate_forecast.web.api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 is default Hugging Face Spaces port
    reload_flag = os.environ.get("RELOAD", "true").lower() == "true"
    uvicorn.run("climate_forecast.web.api:app", host="0.0.0.0", port=port, reload=reload_flag)
