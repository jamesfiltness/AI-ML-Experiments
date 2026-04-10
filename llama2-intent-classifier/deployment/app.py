import uuid
import modal
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# In-memory job store: job_id -> modal FunctionCall
jobs = {}

INTENT_DESCRIPTIONS = {
    "account_update": "Account Update",
    "billing_query": "Billing Query",
    "direct_debit_change": "Direct Debit Change",
    "drainage_issue": "Drainage Issue",
    "general_enquiry": "General Enquiry",
    "hardship_support": "Hardship Support",
    "meter_query": "Meter Query",
    "meter_reading_submission": "Meter Reading Submission",
    "moving_home": "Moving Home",
    "planned_outage_enquiry": "Planned Outage Enquiry",
    "report_leak": "Report Leak",
    "report_low_pressure": "Report Low Pressure",
    "report_no_water": "Report No Water",
    "water_quality_complaint": "Water Quality Complaint",
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Water Intent Classifier</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f0f4f8;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .card {
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 2.5rem;
      width: 100%;
      max-width: 600px;
    }
    h1 {
      font-size: 1.4rem;
      font-weight: 600;
      color: #1a202c;
      margin-bottom: 0.4rem;
    }
    p.subtitle {
      color: #718096;
      font-size: 0.9rem;
      margin-bottom: 1.8rem;
    }
    textarea {
      width: 100%;
      padding: 0.85rem 1rem;
      border: 1.5px solid #e2e8f0;
      border-radius: 8px;
      font-size: 0.95rem;
      font-family: inherit;
      resize: vertical;
      min-height: 100px;
      color: #2d3748;
      transition: border-color 0.15s;
    }
    textarea:focus { outline: none; border-color: #4299e1; }
    button {
      margin-top: 1rem;
      width: 100%;
      padding: 0.85rem;
      background: #3182ce;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;
    }
    button:hover { background: #2b6cb0; }
    button:disabled { background: #90cdf4; cursor: not-allowed; }
    .result {
      margin-top: 1.5rem;
      padding: 1.2rem 1.4rem;
      border-radius: 8px;
      background: #ebf8ff;
      border: 1.5px solid #bee3f8;
    }
    .result-label {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #2b6cb0;
      margin-bottom: 0.3rem;
    }
    .result-intent {
      font-size: 1.1rem;
      font-weight: 600;
      color: #1a365d;
    }
    .result-raw {
      font-size: 0.8rem;
      color: #718096;
      margin-top: 0.3rem;
      font-family: monospace;
    }
    .pending {
      background: #fffff0;
      border-color: #faf089;
    }
    .pending .result-label { color: #975a16; }
    .pending .result-intent { color: #744210; font-size: 0.95rem; font-weight: 500; }
    .error {
      background: #fff5f5;
      border-color: #fed7d7;
    }
    .error .result-label { color: #c53030; }
    .error .result-intent { color: #742a2a; }
    .spinner {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(0,0,0,0.15);
      border-top-color: #975a16;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    .btn-spinner {
      border-color: rgba(255,255,255,0.4);
      border-top-color: white;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .examples {
      margin-top: 1.5rem;
      border-top: 1px solid #e2e8f0;
      padding-top: 1.2rem;
    }
    .examples p {
      font-size: 0.8rem;
      color: #a0aec0;
      margin-bottom: 0.6rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .example-btn {
      display: inline-block;
      margin: 0.25rem 0.3rem 0.25rem 0;
      padding: 0.35rem 0.75rem;
      background: #edf2f7;
      color: #4a5568;
      border: 1px solid #e2e8f0;
      border-radius: 999px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: background 0.15s;
      width: auto;
      font-weight: 400;
    }
    .example-btn:hover { background: #e2e8f0; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Water Intent Classifier</h1>
    <p class="subtitle">Classify customer messages using a fine-tuned Llama 2 model served via Modal.</p>

    <textarea id="message" placeholder="Type a customer message...">My water has come out brown this morning</textarea>
    <button id="classify-btn" onclick="classify()">Classify</button>

    <div id="result" style="display:none"></div>

    <div class="examples">
      <p>Try an example</p>
      <button class="example-btn" onclick="setExample(this)">My water has come out brown this morning</button>
      <button class="example-btn" onclick="setExample(this)">I'd like to update my bank details for direct debit</button>
      <button class="example-btn" onclick="setExample(this)">There's a burst pipe outside my house</button>
      <button class="example-btn" onclick="setExample(this)">No water at all since 6am, what's happening?</button>
      <button class="example-btn" onclick="setExample(this)">I'm struggling to pay my bill this month</button>
      <button class="example-btn" onclick="setExample(this)">We're moving out next week, how do I close my account?</button>
    </div>
  </div>

  <script>
    let pollInterval = null;

    function setExample(btn) {
      document.getElementById("message").value = btn.textContent;
    }

    function stopPolling() {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }

    async function classify() {
      const message = document.getElementById("message").value.trim();
      if (!message) return;

      stopPolling();

      const btn = document.getElementById("classify-btn");
      const resultDiv = document.getElementById("result");

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner btn-spinner"></span>Submitting…';

      resultDiv.className = "result pending";
      resultDiv.innerHTML = `<div class="result-label">Status</div><div class="result-intent">Queued</div>`;
      resultDiv.style.display = "block";

      let jobId;
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        jobId = data.job_id;
      } catch (e) {
        showError(e.message || "Failed to submit job");
        btn.disabled = false;
        btn.textContent = "Classify";
        return;
      }

      btn.innerHTML = '<span class="spinner btn-spinner"></span>Classifying…';

      const startedAt = Date.now();

      pollInterval = setInterval(async () => {
        const elapsed = Math.round((Date.now() - startedAt) / 1000);

        try {
          const res = await fetch(`/poll/${jobId}`);
          const data = await res.json();

          if (data.status === "pending") {
            resultDiv.innerHTML = `
              <div class="result-label">Status</div>
              <div class="result-intent"><span class="spinner"></span>Running&ensp;<span style="color:#b7791f;font-size:0.8rem">${elapsed}s</span></div>
            `;
          } else if (data.status === "done") {
            stopPolling();
            resultDiv.className = "result";
            resultDiv.innerHTML = `
              <div class="result-label">Predicted Intent</div>
              <div class="result-intent">${data.label}</div>
              <div class="result-raw">${data.intent}</div>
            `;
            btn.disabled = false;
            btn.textContent = "Classify";
          } else {
            stopPolling();
            showError(data.error || "Unknown error");
            btn.disabled = false;
            btn.textContent = "Classify";
          }
        } catch (e) {
          stopPolling();
          showError("Polling failed — is the server running?");
          btn.disabled = false;
          btn.textContent = "Classify";
        }
      }, 2000);
    }

    function showError(msg) {
      const resultDiv = document.getElementById("result");
      resultDiv.className = "result error";
      resultDiv.innerHTML = `<div class="result-label">Error</div><div class="result-intent">${msg}</div>`;
      resultDiv.style.display = "block";
    }

    document.getElementById("message").addEventListener("keydown", function(e) {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) classify();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        Predictor = modal.Cls.from_name("water-intent-classifier", "Predictor")
        call = Predictor().predict.spawn(message)
        job_id = str(uuid.uuid4())
        jobs[job_id] = call
        return jsonify({"job_id": job_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/poll/<job_id>")
def poll(job_id):
    call = jobs.get(job_id)
    if not call:
        return jsonify({"error": "Job not found"}), 404

    try:
        intent = call.get(timeout=0)
        del jobs[job_id]
        label = INTENT_DESCRIPTIONS.get(intent, intent.replace("_", " ").title())
        return jsonify({"status": "done", "intent": intent, "label": label})
    except TimeoutError:
        return jsonify({"status": "pending"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)
