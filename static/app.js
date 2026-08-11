(() => {
  const body = document.body;
  const form = document.getElementById("ask-form");
  const questionInput = document.getElementById("question");
  const tabs = Array.from(document.querySelectorAll(".sport-tab"));
  const taglineEl = document.querySelector(".tagline");

  const clearBtn = document.getElementById("clear-btn");

  const loading = document.getElementById("loading");
  const answerSection = document.getElementById("answer-section");
  const answerText = document.getElementById("answer-text");
  const emptyState = document.getElementById("empty-state");
  const errorState = document.getElementById("error-state");

  const SPORT_LABELS = { football: "כדורגל", basketball: "כדורסל" };
  // Kept so a specific message from the API can replace it for one request
  // without becoming the permanent text of the error line.
  const GENERIC_ERROR = errorState.textContent;

  const TAGLINES = {
    "": "שאלו כל שאלה על גמרי ליגת האלופות וגמרי ה-NBA, וקבלו תשובה מבוססת על נתוני הגמרים ההיסטוריים",
    football: "שאלו כל שאלה על גמרי ליגת האלופות, וקבלו תשובה מבוססת על נתוני הגמרים ההיסטוריים",
    basketball: "שאלו כל שאלה על גמרי ה-NBA, וקבלו תשובה מבוססת על נתוני הגמרים ההיסטוריים",
  };

  let currentSport = "";

  function setSport(sport) {
    currentSport = sport;
    body.className = sport ? `sport-${sport}` : "sport-all";
    tabs.forEach((tab) => {
      tab.setAttribute("aria-selected", String(tab.dataset.sport === sport));
    });
    taglineEl.textContent = TAGLINES[sport] ?? TAGLINES[""];
    // A previous answer/sources are for the old sport context -- leaving them
    // visible after switching tabs reads as a stale or wrong result.
    hideResults();
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => setSport(tab.dataset.sport));
  });

  function hideResults() {
    answerSection.hidden = true;
    emptyState.hidden = true;
    errorState.hidden = true;
  }

  // Only offered once there is something to clear, so it never sits there as
  // a live control over an empty box.
  function syncClearButton() {
    clearBtn.hidden = questionInput.value.trim() === "";
  }

  questionInput.addEventListener("input", syncClearButton);
  clearBtn.addEventListener("click", () => {
    questionInput.value = "";
    syncClearButton();
    hideResults();
    questionInput.focus();
  });
  syncClearButton();

  // The model still emits the occasional Markdown bullet or bold run despite
  // the prompt asking for plain prose, and textContent rendered the asterisks
  // literally. Handle the two forms that actually show up -- leading bullets
  // and **bold** -- as structure, and escape everything else.
  function renderAnswer(answer, degraded, offTabSport) {
    const escape = (s) =>
      s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const inline = (s) => escape(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    answerText.innerHTML = "";
    if (degraded) {
      const notice = document.createElement("p");
      notice.className = "answer-line answer-degraded";
      notice.textContent =
        "שירות ניתוח השאלה אינו זמין כרגע, לכן התשובה מבוססת על חיפוש מצומצם. " +
        "שאלות על ספירה, השוואה או «האם אי פעם» עלולות להיות חלקיות.";
      answerText.appendChild(notice);
    }
    for (const raw of answer.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      const bullet = line.match(/^[*-]\s+(.*)$/);
      const p = document.createElement("p");
      // Latin names and scorelines inside RTL Hebrew reorder on screen without
      // isolation, which can flip "1 - 0" as it is displayed.
      p.className = bullet ? "answer-line answer-bullet" : "answer-line";
      p.innerHTML = inline(bullet ? bullet[1] : line);
      answerText.appendChild(p);
    }

    // The answer already explains that the tab, not the corpus, is why there
    // is nothing to report. Being told to switch tabs and then having to do
    // it by hand is the part that would still cost the user the question.
    if (offTabSport) {
      const action = document.createElement("button");
      action.type = "button";
      action.className = "switch-sport-btn";
      action.textContent = `עברו ללשונית ${SPORT_LABELS[offTabSport] ?? offTabSport} ושאלו שוב`;
      action.addEventListener("click", () => {
        setSport(offTabSport);
        const question = questionInput.value.trim();
        if (question) askQuestion(question);
      });
      answerText.appendChild(action);
    }
  }

  // /api/ask still returns `sources`; the UI no longer shows them.
  async function askQuestion(question) {
    hideResults();
    loading.hidden = false;

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, sport: currentSport || null }),
      });

      if (response.status === 404) {
        emptyState.hidden = false;
        return;
      }
      if (!response.ok) {
        // The API says why when it knows why -- an exhausted daily quota is
        // not something to retry in ten seconds, and the generic line
        // invites exactly that.
        const detail = await response
          .json()
          .then((d) => d.detail)
          .catch(() => null);
        if (typeof detail === "string" && detail) {
          errorState.textContent = detail;
          errorState.hidden = false;
          return;
        }
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();
      renderAnswer(data.answer, data.degraded, data.off_tab_sport);
      answerSection.hidden = false;
    } catch (err) {
      console.error(err);
      errorState.textContent = GENERIC_ERROR;
      errorState.hidden = false;
    } finally {
      loading.hidden = true;
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;
    askQuestion(question);
  });
})();
