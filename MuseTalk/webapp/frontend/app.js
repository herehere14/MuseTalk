// app.js (patched)

// If your frontend is served by the same FastAPI backend (recommended),
// use relative URLs so it works via SSH tunnel and also when deployed.
const API = ""; // "" => same origin

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("form");
  const input = document.getElementById("input");
  const log = document.getElementById("log");
  const loading = document.getElementById("loading");
  const avatar = document.getElementById("avatar");

  // Safety: if any element is missing, show what’s wrong and stop.
  if (!form || !input || !log || !loading || !avatar) {
    console.error("Missing DOM elements:", {
      form,
      input,
      log,
      loading,
      avatar,
    });
    return;
  }

  function addMsg(text, who) {
    const div = document.createElement("div");
    div.className = `msg ${who}`;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function setLoading(on) {
    loading.classList.toggle("hidden", !on);
  }

  async function pollStatus(jobId) {
    let shownText = false;

    while (true) {
      const res = await fetch(`${API}/status/${jobId}`);
      const data = await res.json();

      if (data.status === "error") throw new Error(data.error || "Unknown error");

      // show assistant text as soon as ready (only once)
      if (data.reply_text && !shownText) {
        addMsg(data.reply_text, "bot");
        shownText = true;
      }

      if (data.status === "done") return;
      await new Promise((r) => setTimeout(r, 600));
    }
  }

  async function playVideo(jobId) {
    // cache-bust so browser reloads new mp4
    avatar.src = `${API}/video/${jobId}?t=${Date.now()}`;
    avatar.muted = false; // if your mp4 contains audio

    try {
      await avatar.play();
    } catch (e) {
      // Autoplay policies can block audio until user interacts.
      console.warn("Autoplay blocked, muting video and retrying:", e);
      avatar.muted = true;
      await avatar.play();
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = "";

    addMsg(text, "me");
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_text: text }),
      });

      const json = await res.json();
      const job_id = json.job_id;
      if (!job_id) throw new Error("Backend did not return job_id");

      await pollStatus(job_id);
      await playVideo(job_id);
    } catch (err) {
      addMsg(`Error: ${err.message || err}`, "bot");
      console.error(err);
    } finally {
      setLoading(false);
    }
  });
});