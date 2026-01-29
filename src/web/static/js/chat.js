/**
 * Cooking Assistant - Chat Module
 * Handles chat interface, SSE streaming, and conversation management
 */

class ChatApp {
    constructor() {
        this.api = new API('/api');
        this.currentUser = null;
        this.currentConversation = null;
        this.isStreaming = false;

        // DOM Elements
        this.elements = {
            sidebar: document.getElementById('sidebar'),
            sidebarToggle: document.getElementById('sidebarToggle'),
            newChatBtn: document.getElementById('newChatBtn'),
            conversationsList: document.getElementById('conversationsList'),
            loginForm: document.getElementById('loginForm'),
            userInfo: document.getElementById('userInfo'),
            userSelect: document.getElementById('userSelect'),
            loginBtn: document.getElementById('loginBtn'),
            logoutBtn: document.getElementById('logoutBtn'),
            userName: document.getElementById('userName'),
            userAvatar: document.getElementById('userAvatar'),
            welcomeScreen: document.getElementById('welcomeScreen'),
            chatMessages: document.getElementById('chatMessages'),
            chatForm: document.getElementById('chatForm'),
            chatInput: document.getElementById('chatInput'),
            sendBtn: document.getElementById('sendBtn'),
        };

        // Templates
        this.templates = {
            message: document.getElementById('messageTemplate'),
            recipeCard: document.getElementById('recipeCardTemplate'),
            conversation: document.getElementById('conversationTemplate'),
        };

        this.init();
    }

    async init() {
        this.bindEvents();
        await this.loadUsers();
        await this.checkAuth();
    }

    bindEvents() {
        // Sidebar toggle (mobile)
        this.elements.sidebarToggle?.addEventListener('click', () => {
            this.elements.sidebar.classList.toggle('open');
        });

        // New chat button
        this.elements.newChatBtn.addEventListener('click', () => {
            this.startNewConversation();
        });

        // Login
        this.elements.loginBtn.addEventListener('click', () => this.login());
        this.elements.userSelect.addEventListener('change', () => {
            this.elements.loginBtn.disabled = !this.elements.userSelect.value;
        });

        // Logout
        this.elements.logoutBtn.addEventListener('click', () => this.logout());

        // Chat form
        this.elements.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // Chat input auto-resize and keyboard shortcuts
        this.elements.chatInput.addEventListener('input', () => {
            this.autoResizeInput();
        });

        this.elements.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Suggestion chips
        document.querySelectorAll('.chip[data-message]').forEach(chip => {
            chip.addEventListener('click', () => {
                const message = chip.dataset.message;
                this.elements.chatInput.value = message;
                this.sendMessage();
            });
        });

        // Close sidebar on outside click (mobile)
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (!this.elements.sidebar.contains(e.target) &&
                    !this.elements.sidebarToggle.contains(e.target)) {
                    this.elements.sidebar.classList.remove('open');
                }
            }
        });
    }

    autoResizeInput() {
        const input = this.elements.chatInput;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 150) + 'px';
    }

    async loadUsers() {
        try {
            const { users } = await this.api.get('/users');
            this.elements.userSelect.innerHTML = '<option value="">Select a chef...</option>';

            users.forEach(user => {
                const option = document.createElement('option');
                option.value = user.username;
                option.textContent = user.display_name || user.username;
                this.elements.userSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load users:', error);
            Toast.error('Failed to load users');
        }
    }

    async checkAuth() {
        try {
            const { user } = await this.api.get('/auth/whoami');
            this.setUser(user);
            await this.loadConversations();
        } catch (error) {
            if (error.status === 401) {
                this.setUser(null);
            } else {
                console.error('Auth check failed:', error);
            }
        }
    }

    async login() {
        const username = this.elements.userSelect.value;
        if (!username) return;

        try {
            this.elements.loginBtn.disabled = true;
            this.elements.loginBtn.textContent = 'Entering...';

            const { user } = await this.api.post('/auth/login', { username });
            this.setUser(user);
            await this.loadConversations();
            Toast.success(`Welcome, ${user.display_name || user.username}!`);
        } catch (error) {
            console.error('Login failed:', error);
            Toast.error(error.message || 'Login failed');
        } finally {
            this.elements.loginBtn.disabled = false;
            this.elements.loginBtn.textContent = 'Enter Kitchen';
        }
    }

    async logout() {
        try {
            await this.api.post('/auth/logout', {});
            this.setUser(null);
            this.currentConversation = null;
            this.elements.conversationsList.innerHTML = '';
            this.showWelcomeScreen();
            Toast.success('Logged out successfully');
        } catch (error) {
            console.error('Logout failed:', error);
            Toast.error('Logout failed');
        }
    }

    setUser(user) {
        this.currentUser = user;

        if (user) {
            this.elements.loginForm.classList.add('hidden');
            this.elements.userInfo.classList.remove('hidden');
            this.elements.userName.textContent = user.display_name || user.username;
            this.elements.userAvatar.textContent = (user.display_name || user.username)[0].toUpperCase();
            this.elements.chatInput.disabled = false;
            this.elements.sendBtn.disabled = false;
        } else {
            this.elements.loginForm.classList.remove('hidden');
            this.elements.userInfo.classList.add('hidden');
            this.elements.chatInput.disabled = true;
            this.elements.sendBtn.disabled = true;
        }
    }

    async loadConversations() {
        if (!this.currentUser) return;

        try {
            const { conversations } = await this.api.get('/conversations');
            this.renderConversations(conversations);
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    }

    renderConversations(conversations) {
        this.elements.conversationsList.innerHTML = '';

        conversations.forEach(conv => {
            const template = this.templates.conversation.content.cloneNode(true);
            const item = template.querySelector('.conversation-item');

            item.dataset.id = conv.id;
            item.querySelector('.conversation-title').textContent =
                conv.title || 'New conversation';

            if (this.currentConversation === conv.id) {
                item.classList.add('active');
            }

            item.addEventListener('click', () => {
                this.selectConversation(conv.id);
            });

            this.elements.conversationsList.appendChild(template);
        });
    }

    async selectConversation(conversationId) {
        // Update active state
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.toggle('active', item.dataset.id === conversationId);
        });

        this.currentConversation = conversationId;

        // Load messages
        try {
            const { messages } = await this.api.get(`/conversations/${conversationId}/messages`);
            this.showChatArea();
            this.renderMessages(messages);
        } catch (error) {
            console.error('Failed to load messages:', error);
            Toast.error('Failed to load conversation');
        }

        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            this.elements.sidebar.classList.remove('open');
        }
    }

    startNewConversation() {
        this.currentConversation = null;

        // Remove active state from all conversations
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });

        this.showWelcomeScreen();

        // Focus input
        this.elements.chatInput.focus();
    }

    showWelcomeScreen() {
        this.elements.welcomeScreen.classList.remove('hidden');
        this.elements.chatMessages.classList.add('hidden');
        this.elements.chatMessages.innerHTML = '';
    }

    showChatArea() {
        this.elements.welcomeScreen.classList.add('hidden');
        this.elements.chatMessages.classList.remove('hidden');
    }

    renderMessages(messages) {
        this.elements.chatMessages.innerHTML = '';

        messages.forEach(msg => {
            this.addMessageToUI(msg.role, msg.content, msg.meta?.recipe_cards || []);
        });

        this.scrollToBottom();
    }

    addMessageToUI(role, content, cards = [], isStreaming = false) {
        const template = this.templates.message.content.cloneNode(true);
        const message = template.querySelector('.message');

        message.classList.add(role);

        // Avatar
        const avatar = message.querySelector('.message-avatar');
        avatar.textContent = role === 'user' ? '👤' : '🍳';

        // Content
        const textEl = message.querySelector('.message-text');
        textEl.textContent = content;

        if (isStreaming) {
            textEl.classList.add('streaming');
        }

        // Recipe cards
        if (cards.length > 0) {
            const cardsContainer = message.querySelector('.message-cards');
            cards.forEach(card => {
                cardsContainer.appendChild(this.createRecipeCard(card));
            });
        }

        this.elements.chatMessages.appendChild(message);

        return { message, textEl };
    }

    createRecipeCard(cardData) {
        const template = this.templates.recipeCard.content.cloneNode(true);
        const card = template.querySelector('.recipe-card');

        card.querySelector('.recipe-title').textContent = cardData.title;

        const timeEl = card.querySelector('.time-value');
        if (cardData.time_total) {
            timeEl.textContent = `${cardData.time_total} min`;
        } else {
            card.querySelector('.recipe-time').style.display = 'none';
        }

        const ratingEl = card.querySelector('.rating-value');
        if (cardData.rating_avg) {
            ratingEl.textContent = cardData.rating_avg.toFixed(1);
        } else {
            card.querySelector('.recipe-rating').style.display = 'none';
        }

        card.querySelector('.recipe-summary').textContent =
            cardData.one_sentence_summary || '';

        const ingredients = cardData.key_ingredients || [];
        card.querySelector('.ingredients-list').textContent =
            ingredients.slice(0, 6).join(', ');

        const matchEl = card.querySelector('.recipe-match');
        if (cardData.why_match) {
            matchEl.textContent = cardData.why_match;
        } else {
            matchEl.style.display = 'none';
        }

        return card;
    }

    async sendMessage() {
        const message = this.elements.chatInput.value.trim();
        if (!message || this.isStreaming || !this.currentUser) return;

        // Clear input
        this.elements.chatInput.value = '';
        this.elements.chatInput.style.height = 'auto';

        // Show chat area if on welcome screen
        this.showChatArea();

        // Add user message to UI
        this.addMessageToUI('user', message);
        this.scrollToBottom();

        // Start streaming response
        this.isStreaming = true;
        this.elements.sendBtn.disabled = true;

        // Add placeholder assistant message
        const { message: assistantMsg, textEl } = this.addMessageToUI('assistant', '', [], true);
        this.scrollToBottom();

        try {
            await this.streamChat(message, textEl, assistantMsg);
        } catch (error) {
            console.error('Chat error:', error);
            textEl.textContent = 'Sorry, something went wrong. Please try again.';
            textEl.classList.remove('streaming');
            Toast.error(error.message || 'Chat failed');
        } finally {
            this.isStreaming = false;
            this.elements.sendBtn.disabled = false;
            this.elements.chatInput.focus();
        }
    }

    async streamChat(message, textEl, messageEl) {
        const body = JSON.stringify({
            message,
            conversation_id: this.currentConversation,
        });

        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
            credentials: 'same-origin',
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'Chat request failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullContent = '';
        let recipeCards = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process SSE events
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('event:')) {
                    continue; // We'll use the data line with event type
                }

                if (line.startsWith('data:')) {
                    const dataStr = line.slice(5).trim();
                    if (!dataStr) continue;

                    try {
                        const data = JSON.parse(dataStr);

                        // Determine event type from previous line or data structure
                        if (data.content !== undefined) {
                            // Token event
                            fullContent += data.content;
                            textEl.textContent = fullContent;
                            this.scrollToBottom();
                        } else if (data.cards !== undefined) {
                            // Cards event
                            recipeCards = data.cards;
                        } else if (data.conversation_id !== undefined) {
                            // Done event
                            textEl.classList.remove('streaming');

                            // Update conversation ID if new
                            if (!this.currentConversation) {
                                this.currentConversation = data.conversation_id;
                                await this.loadConversations();
                            }

                            // Add recipe cards
                            if (data.meta?.recipe_cards?.length > 0) {
                                recipeCards = data.meta.recipe_cards;
                            }

                            if (recipeCards.length > 0) {
                                const cardsContainer = messageEl.querySelector('.message-cards');
                                recipeCards.forEach(card => {
                                    cardsContainer.appendChild(this.createRecipeCard(card));
                                });
                                this.scrollToBottom();
                            }
                        } else if (data.error !== undefined) {
                            // Error event
                            throw new Error(data.error.message || 'An error occurred');
                        }
                    } catch (e) {
                        if (e.message && !e.message.includes('JSON')) {
                            throw e;
                        }
                        console.warn('Failed to parse SSE data:', dataStr);
                    }
                }
            }
        }

        // Ensure streaming indicator is removed
        textEl.classList.remove('streaming');
    }

    scrollToBottom() {
        requestAnimationFrame(() => {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        });
    }
}

// Export for global use
window.ChatApp = ChatApp;
