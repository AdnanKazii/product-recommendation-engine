const productGrid = document.getElementById("product-grid");
const resultsSection = document.getElementById("results");
const recGrid = document.getElementById("rec-grid");
const queryLabel = document.getElementById("query-label");

function formatProduct(p) {
  const rating = p.avg_review_score !== null ? `★ ${p.avg_review_score.toFixed(1)}` : "no reviews yet";
  const shortId = p.product_id.slice(0, 8) + "…";
  return `
    <div class="card" data-id="${p.product_id}">
      <div class="card-category">${p.category.replaceAll("_", " ")}</div>
      <div class="card-price">$${p.price.toFixed(2)}</div>
      <div class="card-rating">${rating}</div>
      <div class="card-id">${shortId}</div>
    </div>
  `;
}

function renderGrid(container, products) {
  container.innerHTML = products.map(formatProduct).join("");
  container.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => loadRecommendations(card.dataset.id));
  });
}

async function loadCatalog() {
  productGrid.innerHTML = '<p class="loading">Loading products…</p>';
  try {
    const res = await fetch("/products?limit=12");
    if (!res.ok) throw new Error("request failed");
    const products = await res.json();
    renderGrid(productGrid, products);
  } catch (err) {
    productGrid.innerHTML = '<p class="error">Could not load products.</p>';
  }
}

async function loadRecommendations(productId) {
  resultsSection.classList.remove("hidden");
  recGrid.innerHTML = '<p class="loading">Finding recommendations…</p>';
  queryLabel.textContent = productId.slice(0, 8) + "…";
  resultsSection.scrollIntoView({ behavior: "smooth" });

  try {
    const res = await fetch(`/recommendations/${productId}?k=8`);
    if (!res.ok) throw new Error("request failed");
    const data = await res.json();
    if (data.recommendations.length === 0) {
      recGrid.innerHTML = '<p class="empty">No recommendations found for this product.</p>';
      return;
    }
    renderGrid(recGrid, data.recommendations);
  } catch (err) {
    recGrid.innerHTML = '<p class="error">Something went wrong loading recommendations.</p>';
  }
}

loadCatalog();
