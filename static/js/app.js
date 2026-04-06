const API = "";
const PROCESS_INTERVAL = 200; // ms between frame sends

const app = {
    video: null,
    overlay: null,
    ctx: null,
    stream: null,
    running: false,
    processing: false,
    frameTimer: null,
    fpsCounter: { frames: 0, last: performance.now(), value: 0 },
    lastFrame: null,

    // ── Init ──────────────────────────────────────────
    init() {
        this.video = document.getElementById("video");
        this.overlay = document.getElementById("overlay");
        this.ctx = this.overlay.getContext("2d");
        this.loadDB();
    },

    // ── Camera ────────────────────────────────────────
    async toggleCamera() {
        if (this.running) return this.stopCamera();
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            });
            this.video.srcObject = this.stream;
            await this.video.play();
            this.running = true;
            document.getElementById("btn-start").querySelector("span").textContent = "Stop Camera";
            this.resizeOverlay();
            this.video.addEventListener("resize", () => this.resizeOverlay());
            this.frameTimer = setInterval(() => this.captureAndProcess(), PROCESS_INTERVAL);
            toast("Camera started", "success");
        } catch (e) {
            toast("Camera access denied: " + e.message, "error");
        }
    },

    stopCamera() {
        if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
        this.running = false;
        this.stream = null;
        clearInterval(this.frameTimer);
        this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
        document.getElementById("btn-start").querySelector("span").textContent = "Start Camera";
        document.getElementById("faces-list").innerHTML = `<div class="empty-state"><p>Camera stopped.</p></div>`;
    },

    resizeOverlay() {
        this.overlay.width = this.video.videoWidth;
        this.overlay.height = this.video.videoHeight;
    },

    // ── Frame Capture ─────────────────────────────────
    getFrameB64() {
        const vw = this.video.videoWidth;
        const vh = this.video.videoHeight;
        const maxDim = 480;
        const scale = Math.min(maxDim / Math.max(vw, vh), 1);
        const c = document.createElement("canvas");
        c.width = Math.round(vw * scale);
        c.height = Math.round(vh * scale);
        c.getContext("2d").drawImage(this.video, 0, 0, c.width, c.height);
        return c.toDataURL("image/jpeg", 0.65);
    },

    errorCount: 0,

    async captureAndProcess() {
        if (this.processing || !this.running) return;
        if (!this.video.videoWidth) return;
        this.processing = true;
        const frame = this.getFrameB64();
        this.lastFrame = frame;
        try {
            const fd = new FormData();
            fd.append("image", frame);
            const resp = await fetch(API + "/api/process-frame", { method: "POST", body: fd });
            if (!resp.ok) {
                this.errorCount++;
                if (this.errorCount > 30) {
                    clearInterval(this.frameTimer);
                    toast("Server not responding — check if backend is running", "error");
                }
                this.processing = false;
                return;
            }
            this.errorCount = 0;
            const data = await resp.json();
            if (data.loading) {
                document.getElementById("faces-list").innerHTML =
                    `<div class="empty-state"><p>Models loading... please wait</p></div>`;
                this.processing = false;
                return;
            }
            this.drawResults(data);
            this.updateFaceCards(data.faces);
            this.updateStats(data);
        } catch (e) {
            this.errorCount++;
        }
        this.processing = false;
        this.fpsCounter.frames++;
        const now = performance.now();
        if (now - this.fpsCounter.last > 1000) {
            this.fpsCounter.value = Math.round(
                (this.fpsCounter.frames * 1000) / (now - this.fpsCounter.last)
            );
            this.fpsCounter.frames = 0;
            this.fpsCounter.last = now;
            document.getElementById("fps-value").textContent = this.fpsCounter.value;
        }
    },

    // ── Drawing ───────────────────────────────────────
    drawResults(data) {
        const ctx = this.ctx;
        const w = this.overlay.width;
        const h = this.overlay.height;
        ctx.clearRect(0, 0, w, h);

        for (const face of data.faces) {
            const [x1, y1, x2, y2] = face.bbox;
            const known = face.name !== "Unknown";
            const color = known ? "#10b981" : "#f59e0b";

            // bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.setLineDash([]);
            this.roundRect(ctx, x1, y1, x2 - x1, y2 - y1, 6);
            ctx.stroke();

            // corner accents
            const cl = 14;
            ctx.lineWidth = 3.5;
            ctx.strokeStyle = color;
            const corners = [
                [x1, y1, cl, 0, 0, cl],
                [x2, y1, -cl, 0, 0, cl],
                [x1, y2, cl, 0, 0, -cl],
                [x2, y2, -cl, 0, 0, -cl],
            ];
            for (const [cx, cy, dx1, dy1, dx2, dy2] of corners) {
                ctx.beginPath();
                ctx.moveTo(cx + dx1, cy + dy1);
                ctx.lineTo(cx, cy);
                ctx.lineTo(cx + dx2, cy + dy2);
                ctx.stroke();
            }

            // label background
            const label = known ? face.name : "Unknown";
            ctx.font = "600 13px Inter, sans-serif";
            const tw = ctx.measureText(label).width;
            const lh = 22;
            ctx.fillStyle = color;
            this.roundRect(ctx, x1, y1 - lh - 4, tw + 16, lh, 4);
            ctx.fill();
            ctx.fillStyle = known ? "#000" : "#000";
            ctx.fillText(label, x1 + 8, y1 - 10);

            // emotion + age tag at bottom
            const info = `${face.emotion} · ${face.age}y · ${face.gender}`;
            ctx.font = "500 11px Inter, sans-serif";
            const iw = ctx.measureText(info).width;
            ctx.fillStyle = "rgba(0,0,0,.65)";
            this.roundRect(ctx, x1, y2 + 4, iw + 14, 18, 4);
            ctx.fill();
            ctx.fillStyle = "#fff";
            ctx.fillText(info, x1 + 7, y2 + 17);

            // liveness indicator
            const lx = x2 - 10;
            const ly = y1 - 10;
            ctx.beginPath();
            ctx.arc(lx, ly, 5, 0, Math.PI * 2);
            ctx.fillStyle = face.is_live ? "#10b981" : "#ef4444";
            ctx.fill();
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    },

    roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    },

    // ── Face Info Cards ───────────────────────────────
    updateFaceCards(faces) {
        const container = document.getElementById("faces-list");
        if (!faces.length) {
            container.innerHTML = `<div class="empty-state"><p>No faces in frame</p></div>`;
            return;
        }
        container.innerHTML = faces.map((f, i) => this.faceCardHTML(f, i)).join("");
    },

    faceCardHTML(f, i) {
        const known = f.name !== "Unknown";
        const confClass = f.confidence > 0.65 ? "conf-high" : f.confidence > 0.4 ? "conf-med" : "conf-low";
        const confPct = Math.round(f.confidence * 100);

        const emoColors = {
            happy: "#10b981", sad: "#6366f1", surprise: "#f59e0b",
            angry: "#ef4444", fear: "#a855f7", disgust: "#84cc16", neutral: "#6b7094",
        };

        let emoHTML = "";
        if (f.emotion_scores) {
            const sorted = Object.entries(f.emotion_scores).slice(0, 4);
            emoHTML = `<div class="emotion-bar">${sorted
                .map(([k, v]) => {
                    const c = emoColors[k] || "#6b7094";
                    return `<div class="emo-row"><span class="emo-label">${k}</span><div class="emo-track"><div class="emo-fill" style="width:${Math.round(v * 100)}%;background:${c}"></div></div></div>`;
                })
                .join("")}</div>`;
        }

        let celebHTML = "";
        if (f.celebrity_match) {
            const cm = f.celebrity_match;
            const thumbSrc = cm.thumbnail ? `data:image/jpeg;base64,${cm.thumbnail}` : "";
            celebHTML = `<div class="celeb-match">
                ${thumbSrc ? `<img class="celeb-thumb" src="${thumbSrc}" alt="${cm.name}"/>` : ""}
                <div class="celeb-info"><span class="label">★ ${cm.name}</span><span class="celeb-sim">${Math.round(cm.similarity * 100)}%</span></div>
            </div>`;
        }

        return `<div class="face-card">
            <div class="face-card-header">
                <span class="face-name ${known ? "known" : "unknown"}">${f.name}</span>
                ${known ? `<span class="face-confidence ${confClass}">${confPct}%</span>` : ""}
            </div>
            <div class="face-attrs">
                <div class="face-attr"><span class="face-attr-label">Age</span><span class="face-attr-value">${f.age}</span></div>
                <div class="face-attr"><span class="face-attr-label">Gender</span><span class="face-attr-value">${f.gender}</span></div>
                <div class="face-attr"><span class="face-attr-label">Emotion</span><span class="face-attr-value">${f.emotion}</span></div>
                <div class="face-attr"><span class="face-attr-label">Detection</span><span class="face-attr-value">${Math.round(f.det_score * 100)}%</span></div>
            </div>
            <span class="liveness-badge ${f.is_live ? "live-yes" : "live-no"}">${f.is_live ? "✓ Live" : "✗ Spoof?"} (${Math.round(f.liveness_score * 100)}%)</span>
            ${celebHTML}
            ${emoHTML}
        </div>`;
    },

    updateStats(data) {
        document.getElementById("face-count").textContent = data.count;
        document.getElementById("latency-value").textContent = data.processing_ms + "ms";
    },

    // ── Registration ──────────────────────────────────
    openRegister() {
        if (!this.running) return toast("Start the camera first", "error");
        document.getElementById("register-modal").style.display = "flex";
        const c = document.getElementById("register-preview");
        const cx = c.getContext("2d");
        cx.drawImage(this.video, 0, 0, 200, 200);
        document.getElementById("register-name").focus();
    },

    closeRegister() {
        document.getElementById("register-modal").style.display = "none";
    },

    _registering: false,

    async registerFace() {
        if (this._registering) return;
        const name = document.getElementById("register-name").value.trim();
        if (!name) return toast("Please enter a name", "error");
        if (name.length < 2) return toast("Name must be at least 2 characters", "error");

        this._registering = true;
        const btn = document.querySelector("#register-modal .btn-primary");
        const origText = btn.innerHTML;
        btn.innerHTML = "Registering…";
        btn.disabled = true;

        const frame = this.getFrameB64();
        const fd = new FormData();
        fd.append("name", name);
        fd.append("image", frame);
        try {
            const resp = await fetch(API + "/api/register", { method: "POST", body: fd });
            const data = await resp.json();
            if (resp.ok) {
                toast(data.message || `Registered "${name}" successfully!`, data.updated ? "info" : "success");
                this.closeRegister();
                document.getElementById("register-name").value = "";
                this.loadDB();
            } else {
                toast(data.detail || "Registration failed", "error");
            }
        } catch (e) {
            toast("Network error — check if server is running", "error");
        } finally {
            this._registering = false;
            btn.innerHTML = origText;
            btn.disabled = false;
        }
    },

    // ── Reconstruction ────────────────────────────────
    async showReconstruction() {
        if (!this.running) return toast("Start the camera first", "error");
        document.getElementById("recon-modal").style.display = "flex";
        const oc = document.getElementById("recon-original");
        oc.getContext("2d").drawImage(this.video, 0, 0, 192, 192);
        document.getElementById("recon-sim").textContent = "Processing…";
        document.getElementById("recon-explain").textContent = "";

        const frame = this.getFrameB64();
        const fd = new FormData();
        fd.append("image", frame);
        try {
            const resp = await fetch(API + "/api/reconstruct-face", { method: "POST", body: fd });
            const data = await resp.json();
            if (resp.ok) {
                const img = document.getElementById("recon-result");
                img.src = "data:image/jpeg;base64," + data.image_b64;
                document.getElementById("recon-sim").textContent = Math.round(data.similarity * 100) + "%";
                document.getElementById("recon-explain").innerHTML =
                    `<strong>Method:</strong> ${data.method}<br>` +
                    `<strong>Similarity:</strong> ${Math.round(data.similarity * 100)}%<br><br>` +
                    `The reconstructed image represents the model's internal understanding of this face based on the embedding. ` +
                    `It is <em>not</em> the original image — it reflects what information is preserved in the compact 512-dimensional embedding vector. ` +
                    `Higher similarity indicates more identity information is retained.`;
            } else {
                document.getElementById("recon-sim").textContent = "Error";
                document.getElementById("recon-explain").textContent = data.detail || "Failed";
            }
        } catch (e) {
            document.getElementById("recon-sim").textContent = "Error";
        }
    },

    closeRecon() {
        document.getElementById("recon-modal").style.display = "none";
    },

    // ── Privacy ───────────────────────────────────────
    async checkPrivacy() {
        if (!this.running) return toast("Start the camera first", "error");
        document.getElementById("privacy-modal").style.display = "flex";
        document.getElementById("gauge-label").textContent = "Analyzing…";
        document.getElementById("gauge-fill").style.width = "0%";
        document.getElementById("privacy-details").innerHTML = "";

        const frame = this.getFrameB64();
        const fd = new FormData();
        fd.append("image", frame);
        try {
            const resp = await fetch(API + "/api/privacy-score", { method: "POST", body: fd });
            const data = await resp.json();
            if (resp.ok) {
                const pct = Math.round(data.leakage_score * 100);
                const fill = document.getElementById("gauge-fill");
                fill.style.width = pct + "%";
                const colors = { critical: "#ef4444", high: "#f59e0b", moderate: "#eab308", low: "#10b981", minimal: "#06b6d4" };
                fill.style.background = colors[data.risk_level] || "#6b7094";
                document.getElementById("gauge-label").textContent = `${data.risk_level.toUpperCase()} — ${pct}%`;

                document.getElementById("privacy-details").innerHTML = `
                    <div class="row"><span class="label">Leakage Score</span><span>${pct}%</span></div>
                    <div class="row"><span class="label">Risk Level</span><span>${data.risk_level}</span></div>
                    <div class="row"><span class="label">Embedding Entropy</span><span>${Math.round(data.embedding_entropy * 100)}%</span></div>
                    <div class="row"><span class="label">Uniqueness</span><span>${Math.round(data.uniqueness_score * 100)}%</span></div>
                    <div class="explain">${data.explanation}</div>
                    <div style="margin-top:12px"><strong style="font-size:.75rem;color:var(--muted)">RECOMMENDATIONS</strong>
                    <ul style="margin-top:6px;padding-left:16px;font-size:.8rem;color:var(--muted)">${data.recommendations.map((r) => `<li>${r}</li>`).join("")}</ul></div>
                `;
            } else {
                document.getElementById("gauge-label").textContent = "Error";
            }
        } catch (e) {
            document.getElementById("gauge-label").textContent = "Network error";
        }
    },

    closePrivacy() {
        document.getElementById("privacy-modal").style.display = "none";
    },

    // ── DB ────────────────────────────────────────────
    async loadDB() {
        try {
            const resp = await fetch(API + "/api/faces");
            const data = await resp.json();
            document.getElementById("db-count").textContent = data.count;
            const container = document.getElementById("db-list");
            if (!data.faces.length) {
                container.innerHTML = `<div class="empty-state" style="padding:14px 0"><p style="font-size:.8rem">No faces registered yet.</p></div>`;
                return;
            }
            container.innerHTML = data.faces
                .map(
                    (f) => `<div class="db-entry">
                    <span class="db-entry-name">${f.name}</span>
                    <button onclick="app.deleteFace(${f.id}, '${f.name.replace(/'/g, "\\'")}')">Remove</button>
                </div>`
                )
                .join("");
        } catch (_) {}
    },

    async deleteFace(id, name) {
        if (!confirm(`Remove "${name || id}" from the database?`)) return;
        try {
            const resp = await fetch(API + `/api/faces/${id}`, { method: "DELETE" });
            if (resp.ok) {
                toast("Face removed", "info");
                this.loadDB();
            } else {
                const data = await resp.json();
                toast(data.detail || "Failed to remove", "error");
            }
        } catch (_) {
            toast("Network error", "error");
        }
    },
};

// ── Toast helper ──────────────────────────────────────
function toast(msg, type = "info") {
    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    document.getElementById("toast-container").appendChild(el);
    setTimeout(() => el.remove(), 3500);
}

// ── Keyboard shortcuts ────────────────────────────────
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        app.closeRegister();
        app.closeRecon();
        app.closePrivacy();
    }
});
document.addEventListener("DOMContentLoaded", () => {
    app.init();
    const nameInput = document.getElementById("register-name");
    if (nameInput) {
        nameInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") app.registerFace();
        });
    }
});
