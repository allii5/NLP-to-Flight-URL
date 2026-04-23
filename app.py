
import streamlit as st
import requests

API_URL = "http://localhost:8000/generate-search-url"

st.set_page_config(page_title="Aircraft Assistant", layout="centered")

st.title("✈️ Aircraft AI Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! What aircraft are you looking for?"}
    ]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask about aircraft..."):

    # 1. Display user message instantly
    with st.chat_message("user"):
        st.markdown(prompt)

    
    payload = {
        "query": prompt,
        "history": st.session_state.messages 
    }

    
    st.session_state.messages.append({"role": "user", "content": prompt})

    
    with st.chat_message("assistant"):
        with st.spinner("Searching the marketplace..."):
            try:
                response = requests.post(API_URL, json=payload)

                
                if response.status_code != 200:
                    st.error(f"Backend Error: {response.text}")
                else:
                    data = response.json()

                    if data["is_relevant"]:
                        st.markdown(data["summary"])
                        
                        
                        full_response = data["summary"]

                        if data["url"]:
                            link_text = f"\n\n[🔗 View full results on PlaneFax]({data['url']})"
                            st.markdown(link_text)
                            full_response += link_text # Save the link to memory too

                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response
                        })

                    else:
                        st.error(data["summary"])
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["summary"]
                        })

            except Exception as e:
                st.error(f"Connection Error: Is the FastAPI server running? Details: {e}")