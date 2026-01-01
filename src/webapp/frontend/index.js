let recommendationQueue = [];
let currentUsername = 'test';

document.getElementById('recommendButton').addEventListener('click', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    currentUsername = urlParams.get('username') || 'test';
    const btn = document.getElementById('recommendButton');
    
    btn.disabled = true;
    btn.textContent = 'Loading...';

    try {
        const response = await fetch(`/recommend?username=${currentUsername}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch');
        }

        if (!data.recommendations || data.recommendations.length === 0) {
            document.getElementById('recommendations').innerHTML = '<p>No recommendations found.</p>';
            return;
        }

        recommendationQueue = data.recommendations;
        showNextCard();
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Get Recommendations';
    }
});

function showNextCard() {
    const container = document.getElementById('recommendations');
    container.innerHTML = '';

    if (recommendationQueue.length === 0) {
        container.innerHTML = '<h2>No more recommendations. Click "Get Recommendations" for more.</h2>';
        return;
    }

    const game = recommendationQueue.shift();
    const card = createGameCard(game, currentUsername);
    container.appendChild(card);
}

function createGameCard(game, username) {
    const card = document.createElement('div');
    card.className = 'recommendation-card';

    // Header
    const header = document.createElement('div');
    header.className = 'card-header';
    
    const title = document.createElement('h3');
    title.textContent = game.name;
    
    const price = document.createElement('span');
    price.className = 'price-tag';
    price.textContent = (game.price && game.price !== '0') ? `$${game.price}` : 'Free';
    
    header.append(title, price);

    // Media
    const mediaContainer = document.createElement('div');
    mediaContainer.className = 'media-container';
    
    if (game.movieURLs && game.movieURLs.length > 0) {
        const video = document.createElement('video');
        video.src = game.movieURLs[0];
        video.controls = true;
        video.muted = true; 
        mediaContainer.appendChild(video);
    }
    
    if (game.screenshotURLs) {
        game.screenshotURLs.forEach(url => {
            const img = document.createElement('img');
            img.src = url;
            img.loading = 'lazy';
            mediaContainer.appendChild(img);
        });
    }

    // Tags
    const tagsContainer = document.createElement('div');
    tagsContainer.className = 'tags-container';
    
    const allTags = [...(game.genres || []), ...(game.categories || [])];
    const uniqueTags = [...new Set(allTags)];
    
    uniqueTags.forEach(tag => {
        const span = document.createElement('span');
        span.className = 'tag';
        span.textContent = tag;
        tagsContainer.appendChild(span);
    });

    // Description
    const desc = document.createElement('div');
    desc.className = 'description';
    desc.innerHTML = game.shortDescription || '';

    // Detailed Description
    let details;
    if (game.detailedDescription) {
        details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Read More';
        const content = document.createElement('div');
        content.innerHTML = game.detailedDescription;
        details.append(summary, content);
    }

    // Metadata
    const metadata = document.createElement('div');
    metadata.className = 'metadata';
    
    const reviews = document.createElement('span');
    const pos = parseInt(game.positive) || 0;
    const neg = parseInt(game.negative) || 0;
    const totalReviews = pos + neg;
    
    let reviewText = 'No reviews';
    if (totalReviews > 0) {
        const pct = Math.round((pos / totalReviews) * 100);
        reviewText = `${pct}% Positive (${totalReviews.toLocaleString()} reviews)`;
    }
    reviews.textContent = reviewText;
    
    const age = document.createElement('span');
    age.textContent = (game.requiredAge && game.requiredAge !== '0') ? `Age: ${game.requiredAge}+` : '';

    metadata.append(reviews, age);

    // Feedback Buttons
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = 'feedback-div';

    const wishlistBtn = document.createElement('button');
    wishlistBtn.textContent = 'Wishlist';
    wishlistBtn.className = 'feedback-button';
    wishlistBtn.style.backgroundColor = '#2ecc71';
    wishlistBtn.onclick = () => sendFeedback(game, 'wishlist', card, username);

    const skipBtn = document.createElement('button');
    skipBtn.textContent = 'Skip';
    skipBtn.className = 'feedback-button';
    skipBtn.style.backgroundColor = '#95a5a6';
    skipBtn.onclick = () => sendFeedback(game, 'skip', card, username);

    feedbackDiv.append(wishlistBtn, skipBtn);

    // Assemble
    card.append(header, mediaContainer, tagsContainer, desc);
    if (details) card.append(details);
    card.append(metadata, feedbackDiv);

    return card;
}

async function sendFeedback(game, type, card, username) {
    try {
        const res = await fetch('/feedback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                appID: game.appID,
                feedback: type,
                username: username
            })
        });
        
        if (res.ok) {
            showNextCard();
        } else {
            const d = await res.json();
            alert(d.error || 'Error submitting feedback');
        }
    } catch (e) {
        console.error(e);
        alert('Network error');
    }
}