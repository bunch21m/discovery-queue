// The following code will run on page load 

const recommendationContainer = document.getElementById('recommendations');
const recommendButton = document.getElementById('recommendButton');
// Save configuration
const saveAutoButton = document.getElementById('saveAutoWishlist');
const startAutoButton = document.getElementById('startAutoWishlist');
const stopAutoButton = document.getElementById('stopAutoWishlist');
const autoLimitInput = document.getElementById('autoWishlistLimit');
const statusEl = document.getElementById('autoWishlistStatus');

let recommendationQueue = [];
let currentUsername = 'test';
let currentGame = null;
let currentCardElement = null;
let autoWishlistActive = false;
let autoWishlistTarget = 0;
let autoWishlistProcessed = 0;
let autoWishlistPending = false;
let isFetchingRecommendations = false;
let pendingRecommendationPromise = null;
let pendingFeedbackPromise = null;

// Mapping of comparison operators to symbols for display
const comparisonSymbols = {
    lt: '<',
    lte: '≤',
    gt: '>',
    gte: '≥',
    eq: '='
};

// Used to keep track of the current rule configuration in the UI
// Genres and categories use string rules, others use numeric rules
// Structure:
// ruleConfig = {
//    genres: [{ value: string, score: number }, ...],
//    categories: [{ value: string, score: number }, ...],
//    descriptionKeywords: [{ value: string, score: number }, ...],
//    requiredAge: [{ comparison: string, value: number, score: number }, ...],
//    price: [{ comparison: string, value: number, score: number }, ...],
//    positive: [{ comparison: string, value: number, score: number }, ...],
//    negative: [{ comparison: string, value: number, score: number }, ...],
//    positiveRatio: [{ comparison: string, value: number, score: number }, ...]
// }
const ruleConfig = {
    genres: [],
    categories: [],
    descriptionKeywords: [],
    requiredAge: [],
    price: [],
    positive: [],
    negative: [],
    positiveRatio: []
};

recommendButton.addEventListener('click', () => {
    if (isFetchingRecommendations) {
        return;
    }
    const isAuto = recommendButton.dataset.autoClick === 'true';
    recommendButton.dataset.autoClick = '';
    if (!isAuto) {
        recommendButton.disabled = true;
        recommendButton.textContent = 'Loading...';
    }
    pendingRecommendationPromise = triggerRecommendationFetch(isAuto ? 'auto' : 'user')
        .catch(err => {
            if (isAuto && autoWishlistActive) {
                stopAutoWishlist(`Failed to fetch recommendations: ${err.message}`, true);
            } else {
                alert('Error: ' + err.message);
            }
        })
        .finally(() => {
            if (!isAuto) {
                recommendButton.disabled = false;
                recommendButton.textContent = 'Get Recommendations';
            }
            pendingRecommendationPromise = null;
        });
});

// We need to initialize the rule input handlers
// before we can save the configuration because saving relies on
// the ruleConfig being kept up to date
initializeRuleInputs();
saveAutoButton.addEventListener('click', async () => {
    try {
        await persistAutoWishlistConfig();
        updateAutomationStatus('Configuration saved.');
    } catch (error) {
        updateAutomationStatus(error.message, true);
    }
});

startAutoButton.addEventListener('click', async () => {
    await startAutoWishlist();
});

stopAutoButton.addEventListener('click', () => {
    stopAutoWishlist('Stopped manually.');
});

// End of page load code (although functions below will be defined on page load so I suppose part
// of the code below can be considered part of page load as well)

// Beginning of function definitions
// All code below should be function definitions called from above

/*
 * AUTOMATION LOOP FLOW
 * 
 * 1. Entry Point: startAutoWishlist()
 *    - Initializes state (active flag, counters).
 *    - Calls scheduleAutoWishlistStep() to begin the loop.
 * 
 * 2. Scheduling: scheduleAutoWishlistStep()
 *    - Sets a timeout to call runAutoWishlistStep().
 *    - Provides a delay to prevent UI freezing and allow visual updates.
 * 
 * 3. Execution: runAutoWishlistStep()
 *    - Checks if automation is active.
 *    - If not active, exits early, and resets pending flag. The whole loop stops here.
 *    - If no game is loaded, calls handleAutoQueueDepleted().
 *    - Calls requestDecision(appId) to get score and action (wishlist/skip) from backend.
 *    - Calls sendFeedback() to perform the action.
 *    - Updates progress counters.
 *    - Calls showNextCard() to advance the queue.
 *    - Calls scheduleAutoWishlistStep() to continue the loop.
 * 
 * 4. Queue Management: handleAutoQueueDepleted()
 *    - Calls requestMoreRecommendationsForAutomation() to fetch new games.
 *    - Once fetched, calls scheduleAutoWishlistStep() to resume execution.
 */


// This function sets up event listeners for all rule input blocks
// so that adding/removing rules updates the ruleConfig object accordingly
function initializeRuleInputs() {
    const blocks = document.querySelectorAll('.rule-block');
    blocks.forEach(block => {
        const addBtn = block.querySelector('.add-rule');
        const list = block.querySelector('.rule-list');
        if (addBtn) {
            addBtn.addEventListener('click', () => handleAddRule(block));
        }
        if (list) {
            list.addEventListener('click', event => {
                if (event.target instanceof HTMLElement && event.target.matches('button[data-index]')) {
                    const index = Number(event.target.dataset.index);
                    const type = block.dataset.ruleType;
                    removeRule(type, index);
                }
            });
        }
    });
}

// This function handles adding a new rule from the input fields
// in the given block (in this case, a block might be genres, categories, etc.)
function handleAddRule(block) {
    const type = block.dataset.ruleType;
    const kind = block.dataset.ruleKind;
    if (!type || !kind) {
        return;
    }
    const valueInput = block.querySelector('[data-field="value"]');
    const scoreInput = block.querySelector('[data-field="score"]');
    const comparisonSelect = block.querySelector('[data-field="comparison"]');

    if (!(valueInput instanceof HTMLInputElement) || !(scoreInput instanceof HTMLInputElement)) {
        return;
    }

    const valueRaw = valueInput.value.trim();
    const scoreRaw = scoreInput.value.trim();
    const score = Number(scoreRaw);

    if (!valueRaw) {
        alert('Value is required.');
        return;
    }
    if (!scoreRaw || Number.isNaN(score)) {
        alert('Score must be a number.');
        return;
    }

    if (kind === 'string') {
        ruleConfig[type].push({ value: valueRaw, score });
    } else {
        if (!(comparisonSelect instanceof HTMLSelectElement)) {
            return;
        }
        const numericValue = Number(valueRaw);
        if (Number.isNaN(numericValue)) {
            alert('Numeric value required.');
            return;
        }
        ruleConfig[type].push({
            comparison: comparisonSelect.value,
            value: numericValue,
            score
        });
    }

    valueInput.value = '';
    scoreInput.value = '';
    renderRuleList(type);
}

// This function removes a rule at the given index from the specified type
function removeRule(type, index) {
    if (!type || !Array.isArray(ruleConfig[type])) {
        return;
    }
    ruleConfig[type].splice(index, 1);
    renderRuleList(type);
}

// This function re-renders the rule list for the specified type
// where type is one of the keys in ruleConfig. For example,
// type might be "genres" or "price", and when called with that type,
// it will update the displayed list of rules for that type.
function renderRuleList(type) {
    const block = document.querySelector(`.rule-block[data-rule-type="${type}"]`);
    if (!block) {
        return;
    }
    const list = block.querySelector('.rule-list');
    if (!(list instanceof HTMLElement)) {
        return;
    }
    const entries = ruleConfig[type] || [];
    list.innerHTML = '';
    entries.forEach((entry, index) => {
        const item = document.createElement('li');
        let text = '';
        if ('comparison' in entry) {
            const symbol = comparisonSymbols[entry.comparison] || entry.comparison;
            text = `${symbol} ${entry.value} → ${entry.score}`;
        } else {
            text = `${entry.value} → ${entry.score}`;
        }
        item.textContent = text;
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.dataset.index = String(index);
        item.appendChild(removeButton);
        list.appendChild(item);
    });
}

// This function translates the current ruleConfig object
// into the format expected by the backend endpoint and the auto_wishlister.py functions
function toBackendConfig() {
    return {
        genres: ruleConfig.genres,
        categories: ruleConfig.categories,
        descriptionKeywords: ruleConfig.descriptionKeywords,
        requiredAge: ruleConfig.requiredAge,
        price: ruleConfig.price,
        positive: ruleConfig.positive,
        negative: ruleConfig.negative,
        positiveRatio: ruleConfig.positiveRatio
    };
}

// This function persists the current auto wishlist configuration
// to the backend by sending it to the /auto_wishlist_config endpoint
async function persistAutoWishlistConfig() {
    const payload = {
        username: currentUsername,
        config: toBackendConfig()
    };
    const response = await fetch('/auto_wishlist_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || 'Failed to save configuration.');
    }
    return data;
}

// This function updates the automation status display
// with the given message. If isError is true, it will
// style the message as an error.
function updateAutomationStatus(message, isError = false) {
    statusEl.textContent = message;
    statusEl.classList.toggle('error', isError);
}

// This function enables or disables the automation control buttons
// based on whether the automation is currently running.
function setAutomationControls(running) {
    startAutoButton.disabled = running;
    stopAutoButton.disabled = !running;
    saveAutoButton.disabled = running;
}

// This function starts the auto wishlist process
// and essentially represents the main loop of the automation.
// Associated with the button "Start Auto Wishlist".
async function startAutoWishlist() {
    if (autoWishlistActive) {
        return;
    }

    let limit = Number(autoLimitInput.value);
    if (Number.isNaN(limit)) {
        updateAutomationStatus('Invalid limit value.', true);
        return;
    }
    if (limit === 0) {
        updateAutomationStatus('Limit of 0 means nothing to process.', true);
        return;
    }

    try {
        await persistAutoWishlistConfig();
    } catch (error) {
        updateAutomationStatus(error.message, true);
        return;
    }

    autoWishlistActive = true;
    autoWishlistTarget = limit;
    autoWishlistProcessed = 0;
    updateAutomationStatus('Auto wishlist running...');
    setAutomationControls(true);

    if (!currentGame && recommendationQueue.length === 0) {
        await requestMoreRecommendationsForAutomation();
    } else if (!currentGame && recommendationQueue.length > 0) {
        showNextCard();
    }

    scheduleAutoWishlistStep();
}

// This function stops the auto wishlist process
// and updates the UI accordingly.
// Associated with the button "Stop".
function stopAutoWishlist(message, isError = false) {
    if (!autoWishlistActive && !isError) {
        updateAutomationStatus(message, isError);
        return;
    }
    autoWishlistActive = false;
    autoWishlistPending = false;
    setAutomationControls(false);
    if (message) {
        updateAutomationStatus(message, isError);
    }
}

// This function triggers fetching new recommendations
// from the backend and updates the UI accordingly.
// The source parameter indicates whether the fetch
// was triggered by user action ('user') or automatically
// by the automation process ('auto').
// If we are getting recommendations in batches of 10 games,
// and the auto wishlist is processing 50 games, we want to
// fetch more recommendations when we run out, so this function
// fits into this process by handling the fetch and returning
// the number of recommendations fetched.
async function triggerRecommendationFetch(source) {
    if (isFetchingRecommendations) {
        if (pendingRecommendationPromise) {
            await pendingRecommendationPromise;
            return recommendationQueue.length;
        }
        return recommendationQueue.length;
    }
    isFetchingRecommendations = true;
    try {
        const response = await fetch(`/recommend?username=${currentUsername}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch recommendations.');
        }

        recommendationQueue = (data.recommendations || []).filter(Boolean);
        if (recommendationQueue.length === 0) {
            currentGame = null;
            currentCardElement = null;
            recommendationContainer.innerHTML = '<p>No recommendations found.</p>';
            if (source === 'auto') {
                stopAutoWishlist('No recommendations returned.', true);
            }
            return 0;
        }
        
        showNextCard();
        return recommendationQueue.length;
    } finally {
        isFetchingRecommendations = false;
    }
}

// This function requests more recommendations
// specifically for the automation process when
// it runs out of recommendations to process.
// Essentially it just simulates a click on the
// "Get Recommendations" button and waits
// for the fetch to complete.
async function requestMoreRecommendationsForAutomation() {
    if (!autoWishlistActive) {
        return;
    }
    if (isFetchingRecommendations && pendingRecommendationPromise) {
        await pendingRecommendationPromise;
        return;
    }
    recommendButton.dataset.autoClick = 'true';
    recommendButton.click();
    if (pendingRecommendationPromise) {
        await pendingRecommendationPromise;
    }
}

// This function displays the next game card
// from the recommendation queue in the UI.
// If there are no more recommendations,
// it updates the UI accordingly and
// handles the case where the automation
// process needs more recommendations.
function showNextCard() {
    recommendationContainer.innerHTML = '';

    let game = null;
    while (recommendationQueue.length > 0 && !game) {
        game = recommendationQueue.shift();
    }

    if (!game) {
        currentGame = null;
        currentCardElement = null;
        recommendationContainer.innerHTML = '<h2>No more recommendations. Click "Get Recommendations" for more.</h2>';
        if (autoWishlistActive) {
            handleAutoQueueDepleted();
        }
        return;
    }

    currentGame = game;
    currentCardElement = createGameCard(game, currentUsername);
    recommendationContainer.appendChild(currentCardElement);

    if (autoWishlistActive) {
        scheduleAutoWishlistStep();
    }
}

// This function creates a DOM element representing
// a game recommendation card with all its details
// and feedback buttons.
function createGameCard(game, username) {
    const card = document.createElement('div');
    card.className = 'recommendation-card';
    card.dataset.appId = game.appID;

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
    tagsContainer.className = 'tags-section';
    
    if (game.genres && game.genres.length > 0) {
        const genreRow = document.createElement('div');
        genreRow.className = 'tag-row';
        
        const label = document.createElement('span');
        label.className = 'tag-label';
        label.textContent = 'Genres:';
        genreRow.appendChild(label);
        
        const list = document.createElement('div');
        list.className = 'tag-list';
        
        game.genres.forEach(tag => {
            const span = document.createElement('span');
            span.className = 'tag genre-tag';
            span.textContent = tag;
            list.appendChild(span);
        });
        genreRow.appendChild(list);
        tagsContainer.appendChild(genreRow);
    }

    if (game.categories && game.categories.length > 0) {
        const catRow = document.createElement('div');
        catRow.className = 'tag-row';
        
        const label = document.createElement('span');
        label.className = 'tag-label';
        label.textContent = 'Categories:';
        catRow.appendChild(label);
        
        const list = document.createElement('div');
        list.className = 'tag-list';

        game.categories.forEach(tag => {
            const span = document.createElement('span');
            span.className = 'tag category-tag';
            span.textContent = tag;
            list.appendChild(span);
        });
        catRow.appendChild(list);
        tagsContainer.appendChild(catRow);
    }

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
    const pos = Number.parseInt(game.positive, 10) || 0;
    const neg = Number.parseInt(game.negative, 10) || 0;
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
    wishlistBtn.dataset.action = 'wishlist';
    wishlistBtn.addEventListener('click', () => {
        sendFeedback(game, 'wishlist', card, username);
    });

    const skipBtn = document.createElement('button');
    skipBtn.textContent = 'Skip';
    skipBtn.className = 'feedback-button';
    skipBtn.style.backgroundColor = '#95a5a6';
    skipBtn.dataset.action = 'skip';
    skipBtn.addEventListener('click', () => {
        sendFeedback(game, 'skip', card, username);
    });

    feedbackDiv.append(wishlistBtn, skipBtn);

    // Assemble the card from all these parts
    card.append(header, mediaContainer, tagsContainer, desc);
    if (details) {
        card.append(details);
    }
    card.append(metadata, feedbackDiv);

    return card;
}

// This function schedules the next step of the auto wishlist process
// after a short delay to avoid blocking the UI.
// Other functions will call this when they need to continue the 
// process of auto wishlisting.
function scheduleAutoWishlistStep() {
    if (!autoWishlistActive || autoWishlistPending || !currentGame) {
        return;
    }
    autoWishlistPending = true;
    setTimeout(() => {
        runAutoWishlistStep();
    }, 120);
}

// This function performs a single step of the auto wishlist process.
// It evaluates the current game, makes a decision based on backend
// evaluation, and performs the corresponding action (wishlist or skip).
// It also handles updating the UI and scheduling the next step.
// Other functions will call this as part of the automation loop.
async function runAutoWishlistStep() {
    if (!autoWishlistActive || !currentGame) {
        autoWishlistPending = false;
        return;
    }

    if (autoWishlistTarget !== -1 && autoWishlistProcessed >= autoWishlistTarget) {
        stopAutoWishlist('Target processed.');
        autoWishlistPending = false;
        return;
    }

    try {
        const decision = await requestDecision(currentGame.appID);
        updateAutomationStatus(formatAutomationProgress(decision));

        const button = currentCardElement ? currentCardElement.querySelector(`button[data-action="${decision.action}"]`) : null;
        if (!(button instanceof HTMLButtonElement)) {
            stopAutoWishlist('Unable to find action button.', true);
            return;
        }

        button.click();
        if (pendingFeedbackPromise) {
            const success = await pendingFeedbackPromise;
            if (!success) {
                return;
            }
        }

        autoWishlistProcessed += 1;
    } catch (error) {
        stopAutoWishlist(error.message, true);
        return;
    } finally {
        autoWishlistPending = false;
        if (autoWishlistActive) {
            scheduleAutoWishlistStep();
        }
    }
}

// This function requests an auto wishlist decision
// from the backend for the given app ID.
// It returns the decision object containing
// the action and score.
async function requestDecision(appId) {
    const response = await fetch('/auto_wishlist_decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ appID: appId, username: currentUsername })
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || 'Failed to evaluate auto wishlist decision.');
    }
    return data;
}

// This function formats the automation progress message
// including the current decision and score.
function formatAutomationProgress(decision) {
    const targetText = autoWishlistTarget === -1 ? '∞' : String(autoWishlistTarget);
    const nextIndex = autoWishlistProcessed + 1;
    return `Auto wishlist processing ${nextIndex}/${targetText}: ${decision.action} (score: ${decision.score.toFixed(2)})`;
}

// This function handles the case when the recommendation
// queue is depleted during the auto wishlist process.
// It checks if more recommendations are needed
// and requests them from the backend.
async function handleAutoQueueDepleted() {
    if (!autoWishlistActive) {
        return;
    }
    if (autoWishlistTarget !== -1 && autoWishlistProcessed >= autoWishlistTarget) {
        stopAutoWishlist('Target processed.');
        return;
    }
    await requestMoreRecommendationsForAutomation();
}

// This function sends user feedback for a game
// to the backend and handles the response.
// It updates the UI based on success or failure
// and is used by both manual and automated feedback processes.
async function sendFeedback(game, type, card, username) {
    const operation = (async () => {
        try {
            const res = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    appID: game.appID,
                    feedback: type,
                    username
                })
            });
            const responseData = await res.json();
            if (res.ok) {
                showNextCard();
                return true;
            }
            if (autoWishlistActive) {
                stopAutoWishlist(responseData.error || 'Error submitting feedback.', true);
            } else {
                alert(responseData.error || 'Error submitting feedback');
            }
            return false;
        } catch (error) {
            if (autoWishlistActive) {
                stopAutoWishlist('Network error submitting feedback.', true);
            } else {
                alert('Network error');
            }
            return false;
        } finally {
            pendingFeedbackPromise = null;
        }
    })();

    pendingFeedbackPromise = operation;
    return operation;
}
