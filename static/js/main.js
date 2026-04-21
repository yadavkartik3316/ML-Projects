/* main.js – handles prediction form */

// ── Algorithm selector ────────────────────────────────────────────────
let selectedAlgo = "Linear Regression";
document.querySelectorAll(".algo-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".algo-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    selectedAlgo = btn.dataset.algo;
  });
});

// ── Prediction form ──────────────────────────────────────────────────────────
document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const resultBox = document.getElementById("result-box");
  const errorBox  = document.getElementById("error-box");
  errorBox.classList.add("hidden");

  const payload = {
    algorithm:        selectedAlgo,
    study_hours:      parseFloat(document.getElementById("study_hours").value),
    attendance:       parseFloat(document.getElementById("attendance").value),
    prev_grade:       parseFloat(document.getElementById("prev_grade").value),
    assignment_rate:  parseFloat(document.getElementById("assignment_rate").value),
    health:           parseInt(document.getElementById("health").value),
    extra_activities: parseInt(document.getElementById("extra_activities").value),
    internet_access:  parseInt(document.getElementById("internet_access").value),
    parental_edu:     parseInt(document.getElementById("parental_edu").value),
    family_support:   parseInt(document.getElementById("family_support").value),
  };

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      throw new Error(data.error || "Prediction failed.");
    }

    // Populate result
    const score = data.predicted_score;
    const grade = data.grade;

    document.getElementById("result-score").textContent = score.toFixed(1) + " / 100";
    document.getElementById("result-grade").textContent = "Grade: " + grade;
    document.getElementById("result-bar").style.width   = score + "%";
    document.getElementById("result-msg").textContent   = getMessage(score);
    document.getElementById("result-algo").textContent  = "Model: " + selectedAlgo;

    // Grade colour
    const gradeEl = document.getElementById("result-grade");
    gradeEl.style.color =
      score >= 80 ? "#22c55e" :
      score >= 60 ? "#f59e0b" : "#ef4444";

    document.querySelector(".result-placeholder").classList.add("hidden");
    document.querySelector(".result-content").classList.remove("hidden");

  } catch (err) {
    errorBox.textContent = "⚠️ " + err.message;
    errorBox.classList.remove("hidden");
  }
});

function getMessage(score) {
  if (score >= 90) return "🏆 Outstanding performance! Keep it up!";
  if (score >= 80) return "🌟 Excellent work! You're doing great.";
  if (score >= 70) return "👍 Good performance. A little more effort can push you higher!";
  if (score >= 60) return "📚 Average performance. Focus on weak areas.";
  if (score >= 50) return "⚠️ Below average. Consider seeking extra help.";
  return "❌ Poor performance. Immediate improvement needed.";
}


