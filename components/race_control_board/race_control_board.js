const ROOT_ID = "rcb-root";
const COMPOUND_TEXT = {
  SOFT: "#f8fafc",
  MEDIUM: "#07101a",
  HARD: "#07101a",
  UNKNOWN: "#f8fafc",
};

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function styleAttr(style) {
  return style ? ` style="${style}"` : "";
}

function accentVars(color, soft, line, glow) {
  const accent = color || "#38bdf8";
  return [
    `--rcb-accent:${accent}`,
    `--rcb-accent-soft:${soft || "rgba(56, 189, 248, 0.16)"}`,
    `--rcb-accent-line:${line || "rgba(56, 189, 248, 0.4)"}`,
    `--rcb-accent-glow:${glow || "rgba(56, 189, 248, 0.22)"}`,
  ].join(";");
}

function teamLogo(src, alt, className = "rcb-team-logo") {
  if (!src) {
    return "";
  }
  return `<img class="${className}" src="${esc(src)}" alt="${esc(alt)}">`;
}

function compoundBadge(compound, shortLabel) {
  const tone = compound?.color ?? "#94a3b8";
  const textColor = COMPOUND_TEXT[compound?.name || "UNKNOWN"] || "#f8fafc";
  return `<span class="rcb-compound" style="background:${tone};color:${textColor};">${esc(shortLabel)}</span>`;
}

function polylinePoints(points) {
  return (points || []).map((point) => `${point.x},${point.y}`).join(" ");
}

function renderCircuit(payload) {
  const circuit = payload.circuit || {};
  const points = circuit.path_points || [];
  const markers = circuit.markers || [];
  const width = 760;
  const height = 470;

  const verticalLines = [-150, -100, -50, 0, 50, 100, 150]
    .map(
      (x) =>
        `<line x1="${x}" y1="-110" x2="${x}" y2="110" stroke="#0f1824" stroke-width="1" opacity="0.9"></line>`
    )
    .join("");
  const horizontalLines = [-90, -45, 0, 45, 90]
    .map(
      (y) =>
        `<line x1="-170" y1="${y}" x2="170" y2="${y}" stroke="#0f1824" stroke-width="1" opacity="0.9"></line>`
    )
    .join("");

  const markerSvg = markers
    .map((marker) => {
      const halo = marker.selected
        ? `<circle cx="${marker.x}" cy="${marker.y}" r="13" fill="none" stroke="${marker.team_color || "#38bdf8"}" stroke-width="2.3" opacity="0.84"></circle>`
        : "";
      const radius = marker.selected ? 10 : 7;
      const stroke = marker.selected ? (marker.team_color || "#67e8f9") : "#020617";
      const strokeWidth = marker.selected ? 2.2 : 1.2;
      return `
        <g class="rcb-marker" data-driver="${esc(marker.driver_code)}">
          ${halo}
          <circle cx="${marker.x}" cy="${marker.y}" r="${radius}" fill="${marker.color}" stroke="${stroke}" stroke-width="${strokeWidth}"></circle>
          <text x="${marker.x}" y="${marker.y + 3.4}" text-anchor="middle" fill="${marker.text_color}" font-size="${marker.selected ? 7.2 : 6.5}" font-weight="900">${esc(marker.label)}</text>
        </g>
      `;
    })
    .join("");

  return `
    <div class="rcb-panel">
      <div class="rcb-panel-header">
        <span class="rcb-panel-title">Circuit</span>
        <span class="rcb-panel-kicker">${esc(circuit.mode || "schematic")}</span>
      </div>
      <div class="rcb-map-wrap">
        <svg class="rcb-map" viewBox="-170 -110 340 220" role="img" aria-label="${esc(circuit.name || "Circuit map")}">
          ${verticalLines}
          ${horizontalLines}
          <polyline points="${polylinePoints(points)}" fill="none" stroke="#0d1520" stroke-width="28" stroke-linecap="round" stroke-linejoin="round"></polyline>
          <polyline points="${polylinePoints(points)}" fill="none" stroke="#9aaabc" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round" opacity="0.82"></polyline>
          <polyline points="${polylinePoints(points)}" fill="none" stroke="#050911" stroke-width="13" stroke-linecap="round" stroke-linejoin="round"></polyline>
          <polyline points="${polylinePoints(points)}" fill="none" stroke="#1c2736" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" opacity="0.88"></polyline>
          <text class="rcb-start-label" x="${circuit.start_finish?.x ?? -150}" y="${circuit.start_finish?.y ?? -84}">${esc(circuit.start_finish_label || "START")}</text>
          <circle cx="${circuit.start_finish?.marker_x ?? -120}" cy="${circuit.start_finish?.marker_y ?? -84}" r="3.8" fill="#38bdf8" stroke="#dbeafe" stroke-width="1.1"></circle>
          ${markerSvg}
        </svg>
      </div>
    </div>
  `;
}

function renderTimingRow(row) {
  const teamMarkup = row.team_logo_src
    ? `<span class="rcb-team">${teamLogo(row.team_logo_src, `${row.team || "Team"} logo`)}<span class="rcb-team-text">${esc(row.team_short || row.team || "")}</span></span>`
    : `<span class="rcb-team">${esc(row.team_short || row.team || "")}</span>`;
  return `
    <button class="rcb-timing-row ${row.selected ? "is-selected" : ""}" type="button" data-driver="${esc(row.driver_code)}"${styleAttr(
      row.selected ? accentVars(row.team_color, row.team_color_soft, row.team_color_line) : ""
    )}>
      <span class="rcb-pos">${esc(row.position_label)}</span>
      <span class="rcb-code">${esc(row.driver_code)}</span>
      ${teamMarkup}
      <span>${compoundBadge({ name: row.compound, color: row.compound_color }, row.compound_short || row.compound || "?")}</span>
      <span class="rcb-age">${esc(row.tyre_age_label ?? row.tyre_age ?? "n/a")}</span>
      <span class="rcb-call">${esc(row.call_snippet)}</span>
    </button>
  `;
}

function renderTiming(payload) {
  const timing = payload.timing || {};
  const rows = timing.rows || [];
  const rowsHtml = rows.length
    ? rows.map(renderTimingRow).join("")
    : `<div class="rcb-empty">No timing rows are available for this snapshot.</div>`;

  const note = timing.note
    ? `<div class="rcb-limit-note">${esc(timing.note)}</div>`
    : "";

  return `
    <div class="rcb-panel rcb-timing">
      <div class="rcb-panel-header">
        <span class="rcb-panel-title">Timing</span>
        <span class="rcb-panel-kicker">Order panel</span>
      </div>
      <div class="rcb-timing-head">
        <span>Pos</span>
        <span>Drv</span>
        <span>Team</span>
        <span>Tyre</span>
        <span>Age</span>
        <span>Call</span>
      </div>
      <div class="rcb-timing-rows">${rowsHtml}</div>
      ${note}
    </div>
  `;
}

function renderStints(stints) {
  if (!stints || !stints.length) {
    return `<div class="rcb-note-line">No stint history is attached to this checkpoint yet.</div>`;
  }

  return `
    <div class="rcb-strip-title">Stint timeline</div>
    <div class="rcb-stint-strip">
      ${stints
        .map(
          (stint) => `
            <div class="rcb-stint" style="background:${stint.color || "#94a3b8"};">
              <span class="rcb-stint-label">${esc(stint.label)}</span>
              <span class="rcb-stint-summary">${esc(stint.summary || "")}</span>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderSelectedDriver(payload) {
  const driver = payload.selected_driver || {};
  const note = driver.note ? `<div class="rcb-note-line">${esc(driver.note)}</div>` : "";
  const driverPhoto = driver.driver_photo_src
    ? `<div class="rcb-driver-photo"><img src="${esc(driver.driver_photo_src)}" alt="${esc(driver.driver || "Driver")} photo"></div>`
    : "";
  const driverLogo = teamLogo(driver.team_logo_src, `${driver.team || "Team"} logo`, "rcb-driver-logo");
  return `
    <div class="rcb-panel">
      <div class="rcb-panel-header">
        <span class="rcb-panel-title">Selected car</span>
        <span class="rcb-panel-kicker">Tactical drawer</span>
      </div>
      <div class="rcb-drawer"${styleAttr(accentVars(driver.team_color, driver.team_color_soft, driver.team_color_line, driver.team_color_glow))}>
        <div class="rcb-drawer-top">
          <div class="rcb-driver-stack">
            ${driverPhoto}
            <div class="rcb-driver-copy">
              <div class="rcb-driver-name">${esc(driver.driver || "Unknown")}</div>
              <div class="rcb-driver-team">${driverLogo}<span>${esc(driver.team || "Team n/a")}</span></div>
            </div>
          </div>
          <div>${compoundBadge({ name: driver.compound, color: driver.compound_color }, driver.compound || "?")}</div>
        </div>
        <div class="rcb-metrics">
          <div class="rcb-metric"><span class="rcb-metric-label">Start</span><span class="rcb-metric-value">${esc(driver.start_position_label || "P?")}</span></div>
          <div class="rcb-metric"><span class="rcb-metric-label">Now</span><span class="rcb-metric-value">${esc(driver.current_position_label || "P?")}</span></div>
          <div class="rcb-metric"><span class="rcb-metric-label">Laps left</span><span class="rcb-metric-value">${esc(driver.laps_left_label ?? driver.laps_left ?? "n/a")}</span></div>
          <div class="rcb-metric"><span class="rcb-metric-label">Tyre age</span><span class="rcb-metric-value">${esc(driver.tyre_age_label ?? driver.tyre_age ?? "n/a")}</span></div>
          <div class="rcb-metric"><span class="rcb-metric-label">Stint</span><span class="rcb-metric-value">${esc(driver.stint_label || "n/a")}</span></div>
          <div class="rcb-metric"><span class="rcb-metric-label">Track</span><span class="rcb-metric-value">${esc(driver.track_status_label || "n/a")}</span></div>
        </div>
        <div class="rcb-call-card">
          <div class="rcb-call-label">Current pit call</div>
          <div class="rcb-call-title">${esc(driver.call_title || "Overlay unavailable")}</div>
          <div class="rcb-call-subtitle">${esc(driver.call_subtitle || "")}</div>
          ${driver.call_window ? `<div class="rcb-call-window">${esc(driver.call_window)}</div>` : ""}
        </div>
        ${renderStints(driver.stints || [])}
        <div class="rcb-summary-title">Support summary</div>
        <div class="rcb-summary-grid">
          <div class="rcb-summary-cell"><span class="rcb-summary-label">Support</span><span class="rcb-summary-value">${esc(driver.support || "n/a")}</span></div>
          <div class="rcb-summary-cell"><span class="rcb-summary-label">Confidence</span><span class="rcb-summary-value">${esc(driver.confidence || "Pending")}</span></div>
          <div class="rcb-summary-cell"><span class="rcb-summary-label">Risk notes</span><span class="rcb-summary-value">${esc(driver.risk_notes_label || "0")}</span></div>
        </div>
        <div class="rcb-nearby">
          <div class="rcb-nearby-card"><div class="rcb-nearby-label">Ahead</div><div class="rcb-nearby-value">${esc(driver.ahead || "n/a")}</div></div>
          <div class="rcb-nearby-card"><div class="rcb-nearby-label">Behind</div><div class="rcb-nearby-value">${esc(driver.behind || "n/a")}</div></div>
        </div>
        ${note}
      </div>
    </div>
  `;
}

function renderBoard(payload) {
  const meta = payload.meta || {};
  const selected = payload.selected_driver || {};
  const tags = (meta.tags || []).map((tag) => `<span class="rcb-tag">${esc(tag)}</span>`).join("");
  return `
    <section class="rcb-board"${styleAttr(accentVars(selected.team_color, selected.team_color_soft, selected.team_color_line, selected.team_color_glow))}>
      <div class="rcb-topbar">
        <div>
          <div class="rcb-title">${esc(meta.title || "F1 Strategy Lab")}</div>
          <div class="rcb-subtitle">${esc(meta.subtitle || "")}</div>
        </div>
        <div class="rcb-tags">${tags}</div>
      </div>
      <div class="rcb-shell">
        ${renderCircuit(payload)}
        <div class="rcb-right-stack">
          ${renderTiming(payload)}
          ${renderSelectedDriver(payload)}
        </div>
      </div>
    </section>
  `;
}

export default function(component) {
  const root = component.parentElement.querySelector(`#${ROOT_ID}`) || component.parentElement;

  function updateSelection(driverCode) {
    if (!driverCode) {
      return;
    }
    component.setStateValue("selected_driver", driverCode);
  }

  function bindSelectionHandlers() {
    root.querySelectorAll("[data-driver]").forEach((element) => {
      element.onclick = () => {
        const driverCode = element.getAttribute("data-driver");
        updateSelection(driverCode);
      };
    });
  }

  const payload = component.data || {};
  root.innerHTML = renderBoard(payload);
  bindSelectionHandlers();
}
