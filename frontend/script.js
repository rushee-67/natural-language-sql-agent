const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = document.querySelector("#button-label");
const feedback = document.querySelector("#feedback");

const answerSection = document.querySelector("#answer-section");
const answerText = document.querySelector("#answer-text");
const resultStatus = document.querySelector("#result-status");

const sqlCard = document.querySelector("#sql-card");
const sqlText = document.querySelector("#sql-text");
const copySqlButton = document.querySelector("#copy-sql");

const resultsCard = document.querySelector("#results-card");
const resultsHead = document.querySelector("#results-head");
const resultsBody = document.querySelector("#results-body");
const rowCount = document.querySelector("#row-count");
const emptyResults = document.querySelector("#empty-results");
const truncationNote = document.querySelector("#truncation-note");


function setFeedback(message, type = "loading") {
    feedback.textContent = message;
    feedback.className = `feedback ${type}`;
    feedback.hidden = !message;
}


function setLoading(isLoading) {
    submitButton.disabled = isLoading;
    questionInput.disabled = isLoading;
    form.classList.toggle("is-loading", isLoading);

    buttonLabel.textContent = isLoading
        ? "Processing..."
        : "Ask question →";
}


function getErrorMessage(payload, status) {
    const detail = payload?.detail;

    if (typeof detail === "string") {
        return detail;
    }

    if (Array.isArray(detail)) {
        return detail
            .map((item) => item.msg || "Invalid request")
            .join("; ");
    }

    return `The request failed with status ${status}. Please try again.`;
}


function displayValue(value) {
    if (value === null || value === undefined) {
        return "—";
    }

    if (typeof value === "object") {
        return JSON.stringify(value);
    }

    return String(value);
}


function renderTable(columns, rows) {
    resultsHead.replaceChildren();
    resultsBody.replaceChildren();

    const headerRow = document.createElement("tr");

    columns.forEach((column) => {
        const th = document.createElement("th");
        th.scope = "col";
        th.textContent = column;
        headerRow.appendChild(th);
    });

    resultsHead.appendChild(headerRow);

    rows.forEach((row) => {
        const tr = document.createElement("tr");

        columns.forEach((column) => {
            const td = document.createElement("td");
            td.textContent = displayValue(row[column]);
            tr.appendChild(td);
        });

        resultsBody.appendChild(tr);
    });

    emptyResults.hidden = rows.length > 0;
}


function renderResponse(data) {
    answerSection.hidden = false;

    answerText.textContent =
        data.answer || "The agent did not provide an answer.";

    resultStatus.textContent = data.success
        ? "Completed"
        : "Needs attention";

    resultStatus.className = data.success
        ? "result-status success"
        : "result-status warning";

    // Display the SQL, if the agent executed a query.
    const sql = typeof data.sql === "string" ? data.sql : "";

    sqlText.textContent = sql;
    sqlCard.hidden = !sql;

    const rows = Array.isArray(data.rows) ? data.rows : [];
    const reportedColumns = Array.isArray(data.columns)
        ? data.columns
        : [];

    const columns = reportedColumns.length > 0
        ? reportedColumns
        : (rows.length > 0 ? Object.keys(rows[0]) : []);

    const hasResults = columns.length > 0 || rows.length > 0;

    resultsCard.hidden = !hasResults;

    if (hasResults) {
        renderTable(columns, rows);

        const count = Number.isFinite(data.row_count)
            ? data.row_count
            : rows.length;

        rowCount.textContent =
            `${count} row${count === 1 ? "" : "s"} returned`;

        truncationNote.hidden = !data.truncated;
    }
}


form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        setFeedback("Please enter a question first.", "error");
        questionInput.focus();
        return;
    }

    answerSection.hidden = true;
    setLoading(true);
    setFeedback(
        "Inspecting the schema, validating SQL, and processing your question...",
        "loading"
    );

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(getErrorMessage(data, response.status));
        }

        renderResponse(data);

        if (data.success) {
            setFeedback("Your question has been processed.", "success");
        } else {
            setFeedback(
                "The agent returned a response but did not complete the query successfully.",
                "warning"
            );
        }
    } catch (error) {
        setFeedback(
            error.message || "Something went wrong. Please try again.",
            "error"
        );
    } finally {
        setLoading(false);
    }
});


// Populate the input when an example is selected.
document.querySelectorAll(".example-button").forEach((button) => {
    button.addEventListener("click", () => {
        questionInput.value = button.dataset.question || "";
        questionInput.focus();
    });
});


// Copy the generated SQL to the clipboard.
copySqlButton.addEventListener("click", async () => {
    const query = sqlText.textContent;

    if (!query) {
        return;
    }

    try {
        await navigator.clipboard.writeText(query);
        setFeedback("SQL copied to clipboard.", "success");
    } catch {
        setFeedback(
            "Clipboard access failed. Select and copy the SQL manually.",
            "warning"
        );
    }
});