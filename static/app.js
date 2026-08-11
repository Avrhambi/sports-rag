(() => {
  const body = document.body;
  const form = document.getElementById("ask-form");
  const questionInput = document.getElementById("question");
  const tabs = Array.from(document.querySelectorAll(".sport-tab"));
  const taglineEl = document.querySelector(".tagline");

  const loading = document.getElementById("loading");
  const answerSection = document.getElementById("answer-section");
  const answerText = document.getElementById("answer-text");
  const sourcesSection = document.getElementById("sources-section");
  const sourcesList = document.getElementById("sources-list");
  const emptyState = document.getElementById("empty-state");
  const errorState = document.getElementById("error-state");

  const SPORT_LABELS = { football: "כדורגל", basketball: "כדורסל" };

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
    sourcesSection.hidden = true;
    emptyState.hidden = true;
    errorState.hidden = true;
  }

  // The model still emits the occasional Markdown bullet or bold run despite
  // the prompt asking for plain prose, and textContent rendered the asterisks
  // literally. Handle the two forms that actually show up -- leading bullets
  // and **bold** -- as structure, and escape everything else.
  function renderAnswer(answer) {
    const escape = (s) =>
      s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const inline = (s) => escape(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    answerText.innerHTML = "";
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
  }

  function renderSources(sources) {
    sourcesList.innerHTML = "";
    sources.forEach((source, i) => {
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = "sub-tile";
      tile.setAttribute("aria-expanded", "false");
      tile.innerHTML = `
        <span class="tile-number">${String(i + 1).padStart(2, "0")}</span>
        <span>${source.source_title}</span>
        <span class="tile-sport">${SPORT_LABELS[source.sport] ?? source.sport}</span>
      `;

      const panel = document.createElement("div");
      panel.className = "sub-tile-panel";
      panel.hidden = true;
      panel.innerHTML = `<p>${source.text}</p><a href="${source.url}" target="_blank" rel="noopener noreferrer">${source.url}</a>`;

      tile.addEventListener("click", () => {
        const expanded = tile.getAttribute("aria-expanded") === "true";
        tile.setAttribute("aria-expanded", String(!expanded));
        panel.hidden = expanded;
      });

      sourcesList.append(tile, panel);
    });
    sourcesSection.hidden = sources.length === 0;
  }

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
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();
      renderAnswer(data.answer);
      answerSection.hidden = false;
      renderSources(data.sources);
    } catch (err) {
      console.error(err);
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
