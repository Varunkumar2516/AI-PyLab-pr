# app.py
import streamlit as st
from backend import ask_chatbot, summarize_text, creative_writer, make_notes, generate_ideas

st.set_page_config(page_title="GenAI Multi-Tool App", layout="wide")

st.title(" GenAI Multi-Tool App")

# Sidebar for navigation
section = st.sidebar.selectbox("Choose a tool:", 
                           ["Chatbot", "Summarizer", "Creative Writer", "Note Maker", "Idea Generator"])

# Chatbot Section
if section == "Chatbot":
    st.header("💬 Chatbot")
    user_input = st.text_area("Ask me anything:")
    if st.button("Generate Response"):
        if user_input.strip():
            with st.spinner("Generating..."):
                result = ask_chatbot(user_input)
            st.success("Response:")
            st.write(result)

# Summarizer Section
elif section == "Summarizer":
    st.header("📰 Summarizer")
    text_input = st.text_area("Paste text to summarize:")
    if st.button("Summarize"):
        if text_input.strip():
            with st.spinner("Summarizing..."):
                result = summarize_text(text_input)
            st.success("Summary:")
            st.write(result)

# Creative Writer Section
elif section == "Creative Writer":
    st.header("✍️ Creative Writer")
    prompt_input = st.text_area("Enter topic or idea:")
    if st.button("Generate Story/Poem"):
        if prompt_input.strip():
            with st.spinner("Generating..."):
                result = creative_writer(prompt_input)
            st.success("Generated Content:")
            st.write(result)

# Note Maker Section
elif section == "Note Maker":
    st.header("📑 Note Maker")
    notes_input = st.text_area("Paste text to make notes:")
    if st.button("Make Notes"):
        if notes_input.strip():
            with st.spinner("Generating notes..."):
                result = make_notes(notes_input)
            st.success("Notes:")
            st.write(result)

# Idea Generator Section
elif section == "Idea Generator":
    st.header("💡 Idea Generator")
    idea_input = st.text_area("Enter topic for ideas:")
    if st.button("Generate Ideas"):
        if idea_input.strip():
            with st.spinner("Generating ideas..."):
                result = generate_ideas(idea_input)
            st.success("Ideas:")
            st.write(result)
