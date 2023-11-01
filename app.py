import streamlit as st
from pathlib import Path
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# User inputs
st.sidebar.title("Database Options")

# Upload .db files and store in session state
if "uploaded_dbs" not in st.session_state:
    st.session_state.uploaded_dbs = {}

uploaded_file = st.sidebar.file_uploader("Upload a .db file", type=['db'], accept_multiple_files=True)
if uploaded_file:
    for file in uploaded_file:
        # Read the contents of the uploaded file
        if file.name not in st.session_state.uploaded_dbs:
            st.session_state.uploaded_dbs[file.name] = file

# Show checkboxes for uploaded .db files
db_uris = {}
for filename, file in st.session_state.uploaded_dbs.items():
    # Save the file to a temporary directory and create a URI
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    db_uris[filename] = f"sqlite:///{tmp_path}"

# Let the user select which database to use
selected_db = None
for dbname, db_uri in db_uris.items():
    if st.sidebar.checkbox(f"Use {dbname}", key=dbname):
        selected_db = db_uri
        break  # only allow one to be selected at a time

if selected_db:
    db_uri = selected_db
else:
    db_uri = None  # or a default URI if you have one

openai_api_key = st.secrets["openai"]["api_key"]


# Setup agent
llm = OpenAI(openai_api_key=openai_api_key, temperature=0, streaming=True)


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri)


db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
