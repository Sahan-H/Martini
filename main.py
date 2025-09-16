import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager
import re
from datetime import datetime
import pytz
tz = pytz.timezone("Asia/Colombo")
# Get current date and time with timezone
now = datetime.now(tz)
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]

# Format nicely
formatted_time = now.strftime("%I:%M %p %z, %A, %B %d, %Y")

# Configuration
DB_PATH = "martini.db"


# Get today's date
CURRENT_DATE = date.today()  # Make this configurable


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def setup_db():
    """Initialize database with tables and sample data"""
    with get_db_connection() as conn:
        c = conn.cursor()

        # Create tables if not exist
        c.execute('''CREATE TABLE IF NOT EXISTS clients
                     (id INTEGER PRIMARY KEY, name TEXT, address TEXT, contact TEXT, group_type TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS appointments
                     (id INTEGER PRIMARY KEY, client_id INTEGER, date TEXT, time TEXT, service_type TEXT, staff_id INTEGER, duration INTEGER)''')
        c.execute('''CREATE TABLE IF NOT EXISTS staff
                     (id INTEGER PRIMARY KEY, name TEXT, availability TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS invoices
                     (id INTEGER PRIMARY KEY, client_id INTEGER, amount REAL, status TEXT, date TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS alerts
                     (id INTEGER PRIMARY KEY, client_id INTEGER, type TEXT, notes TEXT)''')

        # Insert sample data with error handling
        sample_data = [
            ("clients", [(1, 'Anne Darvin', '123 Main Street', 'anne@example.com', 'Regular Clients'),
                         (2, 'Bob Smith', '456 Elm St', 'bob@example.com', 'Regular Clients'),
                         (3, 'Charlie Brown', '789 Oak Ave', 'charlie@example.com', 'One-Time Jobs'),
                         (4, 'Dana Lee', '101 Pine St Downtown', 'dana@example.com', 'Regular Clients'),
                         (5, 'Eve Frank', '202 Birch Rd Downtown', 'eve@example.com', 'One-Time Jobs'),
                         (6, 'Frank Green', '303 Cedar Ln', 'frank@example.com', 'One-Time Jobs')]),

            ("staff", [(1, 'David Clean', 'Mon-Fri 9-5'),
                       (2, 'Eve Polish', 'Weekends'),
                       (3, 'Grace Help', 'Mon-Wed 10-4')]),

            ("appointments", [(1, 1, '2025-09-20', '10:00', 'Standard Clean', 1, 2),
                              (2, 2, '2025-09-21', '14:00', 'Deep Clean', None, 3),
                              (3, 3, '2025-09-22', '09:00', 'Quick Clean', None, 1),
                              (4, 4, '2025-09-23', '11:00', 'Standard Clean', 2, 2),
                              (5, 5, '2025-09-24', '15:00', 'Deep Clean', None, 4)]),

            ("invoices", [(1, 1, 100.0, 'outstanding', '2025-09-01'),
                          (2, 2, 120.0, 'outstanding', '2025-09-02'),
                          (3, 3, 150.0, 'paid', '2025-09-05'),
                          (4, 4, 90.0, 'outstanding', '2025-09-06'),
                          (5, 5, 200.0, 'paid', '2025-09-07')]),

            ("alerts", [(1, 2, 'High Demand', 'Needs assignment'),
                        (2, 5, 'New Client', 'Assign staff in high-demand area')])
        ]

        for table, data in sample_data:
            for row in data:
                try:
                    placeholders = ','.join(['?' for _ in row])
                    c.execute(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders})", row)
                except sqlite3.Error as e:
                    st.error(f"Error inserting data into {table}: {e}")

        conn.commit()


def initialize_llm_and_db():
    """Initialize LLM and database connections"""
    if API_KEY == "your-api-key-here":
        st.error("Please set your GROQ_API_KEY environment variable")
        st.stop()

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    return llm, db

import re

def parse_duration(duration_str):
    """Extract numeric duration from a string like '3 hours' or '3'."""
    try:
        # If already an integer or float, return as int
        if isinstance(duration_str, (int, float)):
            return int(duration_str)
        # Convert to string and extract digits
        duration_str = str(duration_str)
        match = re.search(r'\d+', duration_str)
        if match:
            return int(match.group())
        return 2  # Default duration if no digits found
    except (ValueError, TypeError):
        return 2  # Default duration on error

def setup_rag_chain(llm):
    """Setup RAG chain for content Q&A"""
    if 'vectorstore' not in st.session_state:
        try:
            urls = [
                "https://www.revox.io/",
                "https://www.revox.io/work",
                "https://www.revox.io/privacy",
                "https://www.revox.io/careers",
                "http://foxtale.studio/"
            ]
            loader = WebBaseLoader(urls)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embeddings)
            st.session_state.vectorstore = vectorstore
        except Exception as e:
            st.error(f"Error setting up RAG chain: {e}")
            return None

    retriever = st.session_state.vectorstore.as_retriever()

    qa_system_prompt = (
        "You are an assistant for question-answering tasks about Revox.io. "
        "Use the retrieved context to answer concisely. Include a snippet of the relevant section from the website as a source preview."
        "\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "{input}")])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def get_voice_recognition_html():
    """HTML component for voice recognition"""
    return """
    <script>
    function startVoiceRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            alert('Speech recognition not supported in this browser');
            return;
        }

        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onresult = (event) => {
            const text = event.results[0][0].transcript;
            const inputs = window.parent.document.querySelectorAll('input[type="text"]');
            for (let input of inputs) {
                if (input.offsetParent !== null && input.getBoundingClientRect().width > 0) {
                    input.value = text;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    break;
                }
            }
        };

        recognition.onerror = (event) => {
            console.log('Speech recognition error: ' + event.error);
            alert('Speech recognition error: ' + event.error);
        };

        recognition.start();
    }
    </script>
    <button onclick="startVoiceRecognition()" style="padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
        🎤 Speak
    </button>
    """


def generate_insights(db, llm):
    """Generate proactive insights from database"""
    try:
        data_summary = ""
        tables = ["clients", "appointments", "staff", "invoices", "alerts"]

        for table in tables:
            try:
                rows = db.run(f"SELECT * FROM {table} LIMIT 10;")
                data_summary += f"Table {table}:\n{rows}\n\n"
            except Exception as e:
                st.warning(f"Error querying {table}: {e}")

        insight_prompt = f"""
        You are a professional operations analyst for a cleaning service. 
        Analyze the database and provide proactive insights focusing on:

        1. **Unassigned Appointments** – appointments without staff assigned
        2. **Outstanding Invoices** – unpaid client invoices  
        3. **Staff Utilization** – workload distribution
        4. **Client Activity** – upcoming appointments and inactive clients
        5. **Alerts** – issues requiring attention

        Present insights concisely with bullet points and actionable recommendations.
        Current date: {CURRENT_DATE}

        Database data:
        {data_summary}
        """

        return llm.invoke(insight_prompt).content
    except Exception as e:
        return f"Error generating insights: {e}"


def parse_command(command, llm):
    """Parse natural language command into structured data"""
    parse_prompt_template = f"""
    You are a precise JSON parser for database commands. Parse the command into a JSON object with sections for each entity: client, appointment, staff, invoice, or alert. 

    Current date: {CURRENT_DATE}. Use this to resolve relative dates.

    Output ONLY valid JSON, no additional text.

    Example: "Update client Anne Darvin at 123 Main Street email to anne@revox.io"
    Output: {{"client": {{"action": "update", "name": "Anne Darvin", "address": "123 Main Street", "new_contact": "anne@revox.io"}}}}

    Command: {command}
    """

    try:
        response = llm.invoke(parse_prompt_template).content.strip()
        # Clean up response to extract JSON
        if response.startswith('```json'):
            response = response[7:-3]
        elif response.startswith('```'):
            response = response[3:-3]

        return json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse command into valid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Error processing command: {e}")


def commit_form_changes(form_data):
    """Commit form changes to database with proper error handling"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()

            # Helper function to get client ID
            def get_client_id(name, address):
                if not name or not address:
                    return None
                c.execute("SELECT id FROM clients WHERE name=? AND address=?", (name, address))
                result = c.fetchone()
                return result[0] if result else None

            # Helper function to get staff ID
            def get_staff_id(name):
                if not name:
                    return None
                c.execute("SELECT id FROM staff WHERE name=?", (name,))
                result = c.fetchone()
                return result[0] if result else None

            changes_made = []

            # Process each entity type
            for entity, data in form_data.items():
                action = data.get('action')
                if not action or action == 'none':
                    continue

                try:
                    if entity == 'client':
                        if action == 'create' and data.get('name') and data.get('address'):
                            c.execute("INSERT INTO clients (name, address, contact, group_type) VALUES (?, ?, ?, ?)",
                                      (data['name'], data['address'], data.get('contact', ''),
                                       data.get('group_type', '')))
                            changes_made.append(f"Created client: {data['name']}")

                        elif action == 'update' and data.get('name') and data.get('address'):
                            # Build update query dynamically
                            updates = []
                            params = []

                            # Check for contact update
                            if data.get('contact'):
                                updates.append("contact=?")
                                params.append(data['contact'])

                            # Check for group type update
                            if data.get('group_type'):
                                updates.append("group_type=?")
                                params.append(data['group_type'])

                            if updates:
                                params.extend([data['name'], data['address']])
                                c.execute(f"UPDATE clients SET {', '.join(updates)} WHERE name=? AND address=?", params)
                                changes_made.append(f"Updated client: {data['name']}")

                        elif action == 'delete' and data.get('name') and data.get('address'):
                            c.execute("DELETE FROM clients WHERE name=? AND address=?", (data['name'], data['address']))
                            changes_made.append(f"Deleted client: {data['name']}")

                    elif entity == 'appointment':
                        if action == 'create':
                            client_id = get_client_id(data.get('client_name'), data.get('client_address'))
                            if client_id and data.get('date') and data.get('time'):
                                staff_id = get_staff_id(data.get('staff_name')) if data.get('staff_name') else None
                                c.execute(
                                    "INSERT INTO appointments (client_id, date, time, service_type, staff_id, duration) VALUES (?, ?, ?, ?, ?, ?)",
                                    (client_id, data['date'], data['time'], data.get('service_type', ''), staff_id,
                                     data.get('duration', 2)))
                                changes_made.append(f"Created appointment for {data.get('client_name')}")
                            else:
                                st.warning(f"Could not create appointment - missing client or required fields")

                        elif action == 'assign' and data.get('id') and data.get('staff_name'):
                            staff_id = get_staff_id(data['staff_name'])
                            if staff_id:
                                c.execute("UPDATE appointments SET staff_id=? WHERE id=?", (staff_id, data['id']))
                                changes_made.append(f"Assigned {data['staff_name']} to appointment {data['id']}")
                            else:
                                st.warning(f"Could not find staff member: {data['staff_name']}")

                        elif action == 'update' and data.get('id'):
                            updates = []
                            params = []

                            if data.get('date'):
                                updates.append("date=?")
                                params.append(data['date'])
                            if data.get('time'):
                                updates.append("time=?")
                                params.append(data['time'])
                            if data.get('service_type'):
                                updates.append("service_type=?")
                                params.append(data['service_type'])
                            if data.get('duration'):
                                updates.append("duration=?")
                                params.append(data['duration'])

                            if updates:
                                params.append(data['id'])
                                c.execute(f"UPDATE appointments SET {', '.join(updates)} WHERE id=?", params)
                                changes_made.append(f"Updated appointment {data['id']}")

                    elif entity == 'staff':
                        if action == 'create' and data.get('name') and data.get('availability'):
                            c.execute("INSERT INTO staff (name, availability) VALUES (?, ?)",
                                      (data['name'], data['availability']))
                            changes_made.append(f"Created staff member: {data['name']}")

                        elif action == 'update' and data.get('name') and data.get('availability'):
                            c.execute("UPDATE staff SET availability=? WHERE name=?",
                                      (data['availability'], data['name']))
                            changes_made.append(f"Updated staff member: {data['name']}")

                        elif action == 'delete' and data.get('name'):
                            c.execute("DELETE FROM staff WHERE name=?", (data['name'],))
                            changes_made.append(f"Deleted staff member: {data['name']}")

                    elif entity == 'invoice':
                        if action == 'create':
                            client_id = get_client_id(data.get('client_name'), data.get('client_address'))
                            if client_id and data.get('amount') and data.get('status') and data.get('date'):
                                c.execute("INSERT INTO invoices (client_id, amount, status, date) VALUES (?, ?, ?, ?)",
                                          (client_id, data['amount'], data['status'], data['date']))
                                changes_made.append(f"Created invoice for {data.get('client_name')}")

                        elif action == 'update':
                            if data.get('id'):
                                # Update by ID
                                updates = []
                                params = []

                                if data.get('amount') is not None:
                                    updates.append("amount=?")
                                    params.append(data['amount'])
                                if data.get('status'):
                                    updates.append("status=?")
                                    params.append(data['status'])
                                if data.get('date'):
                                    updates.append("date=?")
                                    params.append(data['date'])

                                if updates:
                                    params.append(data['id'])
                                    c.execute(f"UPDATE invoices SET {', '.join(updates)} WHERE id=?", params)
                                    changes_made.append(f"Updated invoice {data['id']}")
                            else:
                                # Update by client name/address
                                client_id = get_client_id(data.get('client_name'), data.get('client_address'))
                                if client_id:
                                    updates = []
                                    params = []

                                    if data.get('amount') is not None:
                                        updates.append("amount=?")
                                        params.append(data['amount'])
                                    if data.get('status'):
                                        updates.append("status=?")
                                        params.append(data['status'])
                                    if data.get('date'):
                                        updates.append("date=?")
                                        params.append(data['date'])

                                    if updates:
                                        params.append(client_id)
                                        c.execute(f"UPDATE invoices SET {', '.join(updates)} WHERE client_id=?", params)
                                        changes_made.append(f"Updated invoice for {data.get('client_name')}")

                    elif entity == 'alert':
                        if action == 'create':
                            client_id = get_client_id(data.get('client_name'), data.get('client_address'))
                            if client_id and data.get('type') and data.get('notes'):
                                c.execute("INSERT INTO alerts (client_id, type, notes) VALUES (?, ?, ?)",
                                          (client_id, data['type'], data['notes']))
                                changes_made.append(f"Created alert for {data.get('client_name')}")

                        elif action == 'update' and data.get('id'):
                            updates = []
                            params = []

                            if data.get('type'):
                                updates.append("type=?")
                                params.append(data['type'])
                            if data.get('notes'):
                                updates.append("notes=?")
                                params.append(data['notes'])

                            if updates:
                                params.append(data['id'])
                                c.execute(f"UPDATE alerts SET {', '.join(updates)} WHERE id=?", params)
                                changes_made.append(f"Updated alert {data['id']}")

                        elif action == 'delete':
                            if data.get('id'):
                                c.execute("DELETE FROM alerts WHERE id=?", (data['id'],))
                                changes_made.append(f"Deleted alert {data['id']}")
                            else:
                                client_id = get_client_id(data.get('client_name'), data.get('client_address'))
                                if client_id:
                                    c.execute("DELETE FROM alerts WHERE client_id=?", (client_id,))
                                    changes_made.append(f"Deleted alerts for {data.get('client_name')}")

                except sqlite3.Error as e:
                    st.error(f"Database error processing {entity}: {e}")
                    continue

            conn.commit()
            return changes_made

    except Exception as e:
        st.error(f"Error committing changes: {e}")
        return []


# Initialize the application
setup_db()
llm, db = initialize_llm_and_db()
rag_chain = setup_rag_chain(llm)

# Setup SQL Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=False)

# Streamlit App
st.title("Martini AI Assistant")
st.write(f"Current date and time: {formatted_time}")

tab1, tab2, tab3, tab4 = st.tabs(["Content Q&A", "Service Summary", "DB Query", "Task Completion"])

with tab1:
    st.header("Ask about Revox.io")
    # st.components.v1.html(get_voice_recognition_html(), height=50)
    question = st.text_input("Type your question:", key="qa_input")

    if question and rag_chain:
        try:
            instructions = """
            You are an expert assistant for Revox.io. 
            Keep answers concise with bullet points when appropriate.
            Use only verified information from documentation.
            Provide examples when relevant.
            """
            prompt = f"{instructions}\n\nQuestion: {question}"
            response = rag_chain.invoke({"input": prompt})

            st.markdown("**Answer:**")
            st.info(response["answer"])

            # if response.get("context"):
            #     with st.expander("Source Snippets"):
            #         for i, doc in enumerate(response["context"][:3]):
            #             snippet = doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content
            #             source = doc.metadata.get("source", "Unknown")
            #             st.write(f"**Source {i + 1}:** {source}")
            #             st.write(snippet)
            #             st.divider()
        except Exception as e:
            st.error(f"Error processing question: {e}")

with tab2:
    st.header("Proactive Insights")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Insights"):
            if 'insights' in st.session_state:
                del st.session_state.insights

    if 'insights' not in st.session_state:
        with st.spinner("Generating insights..."):
            st.session_state.insights = generate_insights(db, llm)

    st.write(st.session_state.insights)

with tab3:
    st.header("Natural Language DB Query")
    # st.components.v1.html(get_voice_recognition_html(), height=50)
    query = st.text_input(
        "Type your query (e.g., 'How many clients from Regular Clients group have outstanding invoices?'):",
        key="db_query_input"
    )

    if query:
        try:
            with st.spinner("Processing query..."):
                instructions = """
                Translate the query into correct SQL and execute it.
                Present results clearly with bullet points or tables as appropriate.
                If the query cannot be executed, explain why.
                """
                prompt = f"{instructions}\n\nQuery: {query}"
                result = agent.invoke({"input": prompt})
                st.write("**Result:**")
                st.write(result['output'])
        except Exception as e:
            st.error(f"Error processing query: {e}")

with tab4:
    st.header("Natural Language Task Command")
    # st.components.v1.html(get_voice_recognition_html(), height=50)

    command = st.text_input(
        "Type your command: (e.g. Hey Martini, can you please find client Anne Darvin at 123 Main Street, update their email to anne@revox.io, and create a new appointment for a deep clean on the 1st of September at 9 am for 3 hours?)",
        placeholder="e.g., 'Update client Anne Darvin email to anne@newaddress.com'",
        key="task_input"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Parse Command", disabled=not command):
            try:
                with st.spinner("Parsing command..."):
                    form_data = parse_command(command, llm)
                    st.session_state.form_data = form_data
                    st.success("Command parsed successfully!")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # Interactive Form for reviewing and editing parsed commands
    st.subheader("Review and Edit Parsed Command")
    form_data = st.session_state.get('form_data', {})

    if form_data:
        with st.form(key="parsed_command_form"):
            st.info("✨ Review and modify the parsed command below before committing:")

            updated_form_data = {}

            # Client Operations
            if 'client' in form_data:
                client_data = form_data['client']
                with st.expander("👤 Client Operations", expanded=True):
                    st.write("**Action:** " + client_data.get('action', '').title())

                    col1, col2 = st.columns(2)
                    with col1:
                        client_name = st.text_input("Client Name", value=client_data.get('name', ''),
                                                    key="form_client_name")
                        client_contact = st.text_input("Contact",
                                                       value=client_data.get('contact', '') or client_data.get(
                                                           'new_contact', ''), key="form_client_contact")
                    with col2:
                        client_address = st.text_input("Address", value=client_data.get('address', ''),
                                                       key="form_client_address")
                        client_group = st.selectbox("Group Type",
                                                    ["Regular Clients", "One-Time Jobs", "VIP Clients"],
                                                    index=0 if not client_data.get('group_type') else
                                                    (["Regular Clients", "One-Time Jobs", "VIP Clients"].index(
                                                        client_data.get('group_type'))
                                                     if client_data.get('group_type') in ["Regular Clients",
                                                                                          "One-Time Jobs",
                                                                                          "VIP Clients"] else 0),
                                                    key="form_client_group")

                    updated_form_data['client'] = {
                        'action': client_data.get('action', ''),
                        'name': client_name,
                        'address': client_address,
                        'contact': client_contact,
                        'group_type': client_group
                    }

            # Appointment Operations
            if 'appointment' in form_data:
                appt_data = form_data['appointment']
                with st.expander("📅 Appointment Operations", expanded=True):
                    st.write("**Action:** " + appt_data.get('action', '').title())

                    if appt_data.get('action') == 'assign':
                        col1, col2 = st.columns(2)
                        with col1:
                            appt_id = st.number_input("Appointment ID", value=appt_data.get('id', 0),
                                                      key="form_appt_id")
                        with col2:
                            staff_name = st.text_input("Staff Name", value=appt_data.get('staff_name', ''),
                                                       key="form_appt_staff")

                        updated_form_data['appointment'] = {
                            'action': 'assign',
                            'id': appt_id,
                            'staff_name': staff_name
                        }
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            appt_client_name = st.text_input("Client Name", value=appt_data.get('client_name', ''),
                                                             key="form_appt_client_name")
                            appt_date = st.date_input("Date",
                                                      value=datetime.strptime(appt_data.get('date', CURRENT_DATE),
                                                                              '%Y-%m-%d').date()
                                                      if appt_data.get('date') else datetime.strptime(CURRENT_DATE,
                                                                                                      '%Y-%m-%d').date(),
                                                      key="form_appt_date")
                            appt_service = st.selectbox("Service Type",
                                                        ["Standard Clean", "Deep Clean", "Quick Clean",
                                                         "Move-out Clean"],
                                                        index=0 if not appt_data.get('service_type') else
                                                        (["Standard Clean", "Deep Clean", "Quick Clean",
                                                          "Move-out Clean"].index(appt_data.get('service_type'))
                                                         if appt_data.get('service_type') in ["Standard Clean",
                                                                                              "Deep Clean",
                                                                                              "Quick Clean",
                                                                                              "Move-out Clean"] else 0),
                                                        key="form_appt_service")
                        with col2:
                            appt_client_address = st.text_input("Client Address",
                                                                value=appt_data.get('client_address', ''),
                                                                key="form_appt_client_address")
                            appt_time = st.time_input("Time",
                                                      value=datetime.strptime(appt_data.get('time', '09:00'),
                                                                              '%H:%M').time()
                                                      if appt_data.get('time') else datetime.strptime('09:00',
                                                                                                      '%H:%M').time(),
                                                      key="form_appt_time")
                            # appt_duration = st.number_input("Duration (hours)", min_value=1, max_value=8,
                            #                                 value=int(appt_data.get('duration', 2)),  # Convert to int
                            #                                 key="form_appt_duration")
                            duration_value = parse_duration(appt_data.get('duration', 2))
                            appt_duration = st.number_input("Duration (hours)", min_value=1, max_value=8,
                                                            value=duration_value,
                                                            key="form_appt_duration")

                        updated_form_data['appointment'] = {
                            'action': appt_data.get('action', ''),
                            'client_name': appt_client_name,
                            'client_address': appt_client_address,
                            'date': appt_date.strftime('%Y-%m-%d'),
                            'time': appt_time.strftime('%H:%M'),
                            'service_type': appt_service,
                            'duration': appt_duration
                        }

            # Staff Operations
            if 'staff' in form_data:
                staff_data = form_data['staff']
                with st.expander("👷 Staff Operations", expanded=True):
                    st.write("**Action:** " + staff_data.get('action', '').title())

                    col1, col2 = st.columns(2)
                    with col1:
                        staff_name = st.text_input("Staff Name", value=staff_data.get('name', ''),
                                                   key="form_staff_name")
                    with col2:
                        staff_availability = st.text_input("Availability",
                                                           value=staff_data.get('availability', '') or staff_data.get(
                                                               'new_availability', ''),
                                                           placeholder="e.g., Mon-Fri 9-5",
                                                           key="form_staff_availability")

                    updated_form_data['staff'] = {
                        'action': staff_data.get('action', ''),
                        'name': staff_name,
                        'availability': staff_availability
                    }

            # Invoice Operations
            if 'invoice' in form_data:
                invoice_data = form_data['invoice']
                with st.expander("💰 Invoice Operations", expanded=True):
                    st.write("**Action:** " + invoice_data.get('action', '').title())

                    col1, col2 = st.columns(2)
                    with col1:
                        invoice_client_name = st.text_input("Client Name", value=invoice_data.get('client_name', ''),
                                                            key="form_invoice_client_name")
                        invoice_amount = st.number_input("Amount ($)", min_value=0.0, value=float(
                            invoice_data.get('amount', 0) or invoice_data.get('new_amount', 0)),
                                                         key="form_invoice_amount")
                    with col2:
                        invoice_client_address = st.text_input("Client Address",
                                                               value=invoice_data.get('client_address', ''),
                                                               key="form_invoice_client_address")
                        invoice_status = st.selectbox("Status",
                                                      ["outstanding", "paid", "pending", "overdue"],
                                                      index=0 if not (invoice_data.get('status') or invoice_data.get(
                                                          'new_status')) else
                                                      (["outstanding", "paid", "pending", "overdue"].index(
                                                          invoice_data.get('status') or invoice_data.get('new_status'))
                                                       if (invoice_data.get('status') or invoice_data.get(
                                                          'new_status')) in ["outstanding", "paid", "pending",
                                                                             "overdue"] else 0),
                                                      key="form_invoice_status")

                    invoice_date = st.date_input("Invoice Date",
                                                 value=datetime.strptime(
                                                     invoice_data.get('date', CURRENT_DATE) or invoice_data.get(
                                                         'new_date', CURRENT_DATE), '%Y-%m-%d').date(),
                                                 key="form_invoice_date")

                    updated_form_data['invoice'] = {
                        'action': invoice_data.get('action', ''),
                        'client_name': invoice_client_name,
                        'client_address': invoice_client_address,
                        'amount': invoice_amount,
                        'status': invoice_status,
                        'date': invoice_date.strftime('%Y-%m-%d')
                    }

            # Alert Operations
            if 'alert' in form_data:
                alert_data = form_data['alert']
                with st.expander("⚠️ Alert Operations", expanded=True):
                    st.write("**Action:** " + alert_data.get('action', '').title())

                    col1, col2 = st.columns(2)
                    with col1:
                        alert_client_name = st.text_input("Client Name", value=alert_data.get('client_name', ''),
                                                          key="form_alert_client_name")
                        alert_type = st.selectbox("Alert Type",
                                                  ["High Demand", "New Client", "Payment Issue", "Scheduling Conflict",
                                                   "Special Requirements"],
                                                  index=0 if not (
                                                              alert_data.get('type') or alert_data.get('new_type')) else
                                                  (["High Demand", "New Client", "Payment Issue", "Scheduling Conflict",
                                                    "Special Requirements"].index(
                                                      alert_data.get('type') or alert_data.get('new_type'))
                                                   if (alert_data.get('type') or alert_data.get('new_type')) in [
                                                      "High Demand", "New Client", "Payment Issue",
                                                      "Scheduling Conflict", "Special Requirements"] else 0),
                                                  key="form_alert_type")
                    with col2:
                        alert_client_address = st.text_input("Client Address",
                                                             value=alert_data.get('client_address', ''),
                                                             key="form_alert_client_address")
                        alert_notes = st.text_area("Notes",
                                                   value=alert_data.get('notes', '') or alert_data.get('new_notes', ''),
                                                   height=100,
                                                   key="form_alert_notes")

                    updated_form_data['alert'] = {
                        'action': alert_data.get('action', ''),
                        'client_name': alert_client_name,
                        'client_address': alert_client_address,
                        'type': alert_type,
                        'notes': alert_notes
                    }

            # Commit button
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                commit_button = st.form_submit_button("💾 Commit All Changes", use_container_width=True, type="primary")

            if commit_button:
                with st.spinner("Committing changes..."):
                    changes = commit_form_changes(updated_form_data)
                    if changes:
                        st.success("🎉 Changes committed successfully!")
                        for change in changes:
                            st.write(f"✅ {change}")
                        # Clear form data after successful commit
                        st.session_state.form_data = {}
                        # st.rerun()
                    else:
                        st.warning("⚠️ No changes were made.")

        # Show raw data option
        with st.expander("🔍 View Raw Parsed Data", expanded=False):
            st.json(form_data)

    else:

        st.info("💡 Parse a command above to see an interactive form for reviewing and editing the changes.")