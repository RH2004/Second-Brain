# Second Brain — Demo Recording & Test Script

Use this step-by-step script to record your demo video or run manual end-to-end tests. The script is structured to show off every requirement, including the **proactive surfacing stretch goal**.

---

## Preparation
1. Ensure your `.env` is set up with your `GEMINI_API_KEY`.
2. Open two terminal windows or browser tabs if you're demonstrating the TUI vs Streamlit. 
3. Run the Streamlit app (recommended for the recording as it is highly visual):
   ```powershell
   python -m streamlit run app.py
   ```

---

## Scenario 1: The thinking session (THINK → SAVE)
*This shows the system acting as a reasoning partner, then compressing the discussion into a structured Markdown note.*

1. **Step 1:** In the chat input, type the following user prompt:
   > "I want to think through a design for a distributed key-value cache. It needs to use consistent hashing to distribute keys across nodes, and we should use a replication factor of 3 for high availability."
2. **Step 2:** Read the assistant's reply. Respond to it to build on the idea:
   > "Yes, and to handle node failures, we should use virtual nodes so the load is distributed evenly when a node goes down."
3. **Step 3:** Save the session. Type:
   > "/save"
   *(Alternatively, click the **💾 Save** button in the sidebar).*
4. **Verification:**
   - The assistant will output: `✅ Note saved: Distributed Key-Value Cache Design` (or similar).
   - It shows the generated tags (e.g., `consistent-hashing`, `caching`, `distributed-systems`) and the file path where the `.md` file is saved.

---

## Scenario 2: Fresh Session Recall (FIND)
*This shows the system finding notes in a clean session with no shared context, and using cross-encoder reranking to cite sources.*

1. **Step 1:** Clear the session to start fresh with zero history in context. 
   - Click the **🗑️ New session** button at the bottom of the sidebar.
2. **Step 2:** Ask a specific question about your caching design:
   > "How did I decide to handle node failures in my distributed cache?"
3. **Verification:**
   - The system retrieves the note from `storage/notes/`.
   - The response synthesizes the answer: *"You decided to use virtual nodes to distribute the load evenly when a node goes down..."*
   - Expand the source card under **📚 Sources** to show the retrieved note content, creation timestamp, and relevance score.

---

## Scenario 3: Proactive Surfacing (Stretch Goal)
*This shows the system proactively suggesting the past note mid-thought without being asked.*

1. **Step 1:** Clear the session again by clicking **🗑️ New session** in the sidebar.
2. **Step 2:** Start a brand new discussion on a related topic. The proactive similarity engine now runs on every user message:
   - **Turn 1 (User):** "I'm starting to design a system that needs to store session tokens."
   - **Turn 2 (User):** "I need the system to scale horizontally and handle thousands of reads per second."
   - **Turn 3 (User):** "To prevent hotspots, I'm thinking about hash rings and replication across servers."
3. **Verification:**
   - As soon as your messages are submitted, a banner will appear at the top of the chat area:
     `💡 RELATED NOTE SURFACED: Distributed Key-Value Cache Design`
   - This proves the background task detected the semantic overlap with your previous cache note without you asking for it!

---

## Scenario 4: Usage analytics (HISTORY)
*This shows the MongoDB analytics engine formatting patterns with zero-token template strings.*

1. **Step 1:** Ask the system about your history:
   > "/history"
   *(Or click the **📜 History** button in the sidebar).*
2. **Verification:**
   - The system immediately prints your access patterns (revisited notes, counts, and tags) using the fast-path history templates.
3. **Step 2:** Try a tag-specific history query:
   > "Do I have notes tagged distributed-systems?"
4. **Verification:**
   - The system queries MongoDB and returns a clean list of notes matching that tag.

---

## Scenario 5: The Honest Miss (No Hallucinations)
*This shows that the system says "nothing found" instead of fabricating answers.*

1. **Step 1:** Ask the system about a topic you have never written about:
   > "What did I write about cooking lasagna?"
2. **Verification:**
   - The system checks the index, filters the candidates (which all score below the `-5.0` cross-encoder threshold), and outputs:
     *"I searched my notes but couldn't find anything that closely matches your question about 'cooking lasagna'. This topic may not have been saved yet."*
