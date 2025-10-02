css = '''
<style>
    /* Import a clean font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

    /* General body styling */
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f0f2f6;
    }

    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }

    .chat-message.user {
        background-color: transparent;
        justify-content: flex-end;
        flex-direction: row-reverse;
    }

    .chat-message.bot {
        background-color: transparent;
        justify-content: flex-start;
    }

    .chat-message .message {
        width: 80%;
        padding: 1rem 1.5rem;
        border-radius: 1.25rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    .chat-message.user .message {
        background-color: #0078FF;
        color: #fff;
    }

    .chat-message.bot .message {
        background-color: #FFFFFF;
        color: #333;
    }

    .chat-message .avatar img {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 1rem;
    }

    /* Style for the Streamlit text input */
    div[data-testid="stTextInput"] > div > div > input {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        border: 1px solid #dfe1e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        color: #333;
        caret-color: #0078FF; /* <<<< THE FIX IS HERE <<<< */
    }

    div[data-testid="stTextInput"] > div > div > input:focus {
        border-color: #0078FF;
        box-shadow: 0 0 0 2px rgba(0, 120, 255, 0.25);
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/007/225/199/non_2x/robot-chat-bot-concept-illustration-vector.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn.pixabay.com/photo/2020/07/01/12/58/icon-5359553_1280.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

