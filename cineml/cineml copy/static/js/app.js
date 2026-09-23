/* ── CineML Frontend App ─────────────────────────────────────── */

const API = {
  recommend: '/api/recommend',
  popular: '/api/popular',
  suggest: '/api/suggestions',
  status: '/api/status',
};
// ── DOM refs ──────────────────────────────────────────────────────
const input        = document.getElementById('movie-input');
const searchBtn    = document.getElementById('search-btn');
const btnText      = document.getElementById('btn-text');
const btnSpinner   = document.getElementById('btn-spinner');
const suggestList  = document.getElementById('suggestions');
const resultsSection = document.getElementById('results-section');
const cardsRow     = document.getElementById('cards-row');
const queriedMovie = document.getElementById('queried-movie');
const resultMovieName = document.getElementById('result-movie-name');
const errorToast   = document.getElementById('error-toast');
const popularGrid  = document.getElementById('popular-grid');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('status-text');
const statMovies   = document.getElementById('stat-movies');

// ── Status check ─────────────────────────────────────────────────
async function checkStatus() {

    try {

        const response = await fetch("/api/status");
        const data = await response.json();

        console.log(data);

        // top-right status
        const statusText =
        document.getElementById("status-text");

        if(statusText){

            statusText.innerText =
            `${data.movie_count} movies ready`;
        }


        // left movie count
        const movieCount =
        document.getElementById("stat-movies");

        if(movieCount){

            movieCount.innerText =
            data.movie_count;
        }


        // status dot color
        const statusDot =
        document.getElementById("status-dot");

        if(statusDot){

            if(data.model_ready){

                statusDot.style.background =
                "#22c55e";

            }else{

                statusDot.style.background =
                "#ef4444";
            }
        }

    }

    catch(error){

        console.log(error);

    }

}

checkStatus();
// ── Popular movies ───────────────────────────────────────────────
async function loadPopular() {
  try {
    const res  = await fetch(API.popular);
    const data = await res.json();
    if (!data.movies || !data.movies.length) {
      popularGrid.innerHTML = '<p style="color:var(--faint);font-size:13px">Add a TMDB API key in .env to see trending movies.</p>';
      return;
    }
    popularGrid.innerHTML = data.movies.map(m => `
      <div class="popular-card" onclick="fillAndSearch('${escHtml(m.title)}')">
        <div class="popular-poster-wrap">
          <img class="popular-poster" src="${m.poster || ''}" alt="${escHtml(m.title)}"
               loading="lazy" onerror="this.style.display='none'" />
        </div>
        <div class="popular-info">
          <div class="popular-title">${escHtml(m.title)}</div>
          <div class="popular-rating">⭐ ${m.rating} &nbsp;·&nbsp; ${m.year}</div>
        </div>
      </div>
    `).join('');
  } catch {
    popularGrid.innerHTML = '';
  }
}

// ── Autocomplete ─────────────────────────────────────────────────
let suggestTimer;
input.addEventListener('input', () => {
  clearTimeout(suggestTimer);
  const q = input.value.trim();
  if (q.length < 2) { hideSuggestions(); return; }
  suggestTimer = setTimeout(() => fetchSuggestions(q), 250);
});

async function fetchSuggestions(q) {
  try {
    const res   = await fetch(`${API.suggest}?q=${encodeURIComponent(q)}`);
    const items = await res.json();
    if (!items.length) { hideSuggestions(); return; }
    suggestList.innerHTML = items.map(t =>
      `<li onclick="fillAndSearch('${escHtml(t)}')">${escHtml(t)}</li>`
    ).join('');
    suggestList.classList.remove('hidden');
  } catch { hideSuggestions(); }
}

function hideSuggestions() { suggestList.classList.add('hidden'); }

document.addEventListener('click', e => {
  if (!e.target.closest('.search-wrap')) hideSuggestions();
});

input.addEventListener('keydown', e => {
  if (e.key === 'Enter') recommend();
});

// ── Recommend ────────────────────────────────────────────────────
searchBtn.addEventListener('click', recommend);

function fillAndSearch(title) {
  input.value = title;
  hideSuggestions();
  recommend();
}

async function recommend() {
  const title = input.value.trim();
  if (!title) { showError('Please type a movie name.'); return; }

  setLoading(true);
  hideSuggestions();
  hideError();

  try {
    const res  = await fetch(API.recommend, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ movie: title }),
    });
    const data = await res.json();

    if (data.error) { showError(data.error); setLoading(false); return; }

    renderResults(data);
  } catch (e) {
    showError('Could not reach the server. Is app.py running?');
  }

  setLoading(false);
}

// ── Render ───────────────────────────────────────────────────────
function renderResults(data) {
  const { query, recommendations } = data;

  // queried movie header
  queriedMovie.innerHTML = `
    <img class="queried-poster" src="${query.poster}" alt="${escHtml(query.title)}"
         onerror="this.src=''" />
    <div class="queried-info">
      <h3>${escHtml(query.title)}</h3>
      <div class="meta">${query.year ? query.year + ' &nbsp;·&nbsp; ' : ''}${query.rating !== 'N/A' ? '⭐ ' + query.rating : ''}${query.genres.length ? ' &nbsp;·&nbsp; ' + query.genres.join(', ') : ''}</div>
      ${query.overview ? `<div class="meta" style="margin-top:4px;font-size:11px;color:var(--faint);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${escHtml(query.overview)}</div>` : ''}
    </div>
  `;

  resultMovieName.textContent = query.title;

  cardsRow.innerHTML = recommendations.map((m, i) => `
    <div class="movie-card" style="animation-delay:${i * 80}ms"
         onclick="fillAndSearch('${escHtml(m.title)}')">
      <div class="card-poster-wrap">
        <img class="card-poster" src="${m.poster}" alt="${escHtml(m.title)}"
             loading="lazy" onerror="this.src=''" />
        <div class="card-overlay">
          <div class="card-badge">#${i + 1}</div>
          <div class="card-match">${m.similarity}% match</div>
        </div>
      </div>
      <div class="card-info">
        <div class="card-title">${escHtml(m.title)}</div>
        <div class="card-meta">
          <span>${m.year || '—'}</span>
          <span class="card-rating">${m.rating !== 'N/A' ? '⭐ ' + m.rating : ''}</span>
        </div>
        <div class="card-genres">
          ${m.genres.map(g => `<span class="genre-tag">${escHtml(g)}</span>`).join('')}
        </div>
        <div class="card-bar-wrap">
          <div class="card-bar" style="width:${m.similarity}%"></div>
        </div>
      </div>
    </div>
  `).join('');

  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── UI helpers ────────────────────────────────────────────────────
function setLoading(on) {
  searchBtn.disabled = on;
  btnText.classList.toggle('hidden', on);
  btnSpinner.classList.toggle('hidden', !on);
}

let errorTimer;
function showError(msg) {
  errorToast.textContent = '⚠️  ' + msg;
  errorToast.classList.remove('hidden');
  clearTimeout(errorTimer);
  errorTimer = setTimeout(hideError, 4500);
}
function hideError() { errorToast.classList.add('hidden'); }

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Init ─────────────────────────────────────────────────────────
checkStatus();
loadPopular();
