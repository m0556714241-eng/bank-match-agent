-- =====================================================================
-- מסד "מערכת הנהלת החשבונות" + הזיכרון של הסוכן
-- החלטות: ה-30 (רק מוסיפים, יתרה מחושבת), ה-31 (כסף = מספר שלם ביחידה הקטנה),
--          ה-33 (רישום כיחידה אחת), ה-37 (מבנה הטבלאות)
-- STRICT = המסד בודק סוג של כל ערך (מספר שלם לא יכול להפוך לטקסט בטעות)
--
-- סיומת _minor = סכום כסף כמספר שלם ביחידה הקטנה של המטבע (minor unit):
-- אגורות לשקל, סנטים לדולר ולאירו. למשל 500000 = 5,000.00. לתצוגה מחלקים ב-100.
-- =====================================================================


-- ---------- 1. לקוחות (נטען מהקובץ) ----------
CREATE TABLE customers (
    customer_id       TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    currency          TEXT NOT NULL CHECK (currency IN ('ILS', 'USD', 'EUR')),
    withholding_rate  TEXT NOT NULL            -- שיעור (לא כסף) - נשמר כטקסט בדיוק מלא
) STRICT;


-- ---------- 2. חשבוניות (נטען מהקובץ, לא משתנה לעולם) ----------
CREATE TABLE invoices (
    invoice_id    TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers (customer_id),
    issue_date    TEXT NOT NULL,               -- YYYY-MM-DD
    amount_minor  INTEGER NOT NULL CHECK (amount_minor > 0),
    currency      TEXT NOT NULL CHECK (currency IN ('ILS', 'USD', 'EUR'))
) STRICT;


-- ---------- 3. תקבולים (רק מוסיפים) ----------
CREATE TABLE receipts (
    receipt_id    TEXT PRIMARY KEY,
    tx_id         TEXT NOT NULL UNIQUE,        -- הגנה 1: תנועת בנק לא נרשמת פעמיים
    customer_id   TEXT NOT NULL REFERENCES customers (customer_id),
    amount_minor  INTEGER NOT NULL CHECK (amount_minor > 0),
    currency      TEXT NOT NULL CHECK (currency IN ('ILS', 'USD', 'EUR')),
    certainty     TEXT NOT NULL CHECK (certainty IN ('CERTAIN', 'PROBABLE', 'EXCEPTION')),
    approved_by   TEXT NOT NULL,
    approved_at   TEXT NOT NULL,               -- תאריך ושעה
    run_id        TEXT NOT NULL
) STRICT;


-- ---------- 4. החלות: תקבול <-> חשבונית (רק מוסיפים) ----------
CREATE TABLE receipt_applications (
    application_id  INTEGER PRIMARY KEY,       -- מספר רץ אוטומטי
    receipt_id      TEXT REFERENCES receipts (receipt_id),
    invoice_id      TEXT NOT NULL REFERENCES invoices (invoice_id),
    applied_minor   INTEGER NOT NULL CHECK (applied_minor <> 0),  -- שלילי = ביטול (רשומה הפוכה)
    currency        TEXT NOT NULL CHECK (currency IN ('ILS', 'USD', 'EUR')),
    source          TEXT NOT NULL CHECK (source IN ('AGENT', 'OPENING_BALANCE', 'REVERSAL')),
    -- יתרת פתיחה היא היחידה שאין לה תקבול (הגיעה מהמערכת הקודמת)
    CHECK ((source = 'OPENING_BALANCE') = (receipt_id IS NULL))
) STRICT;

CREATE INDEX idx_applications_invoice ON receipt_applications (invoice_id);


-- ---------- 5. הפרשים (רק מוסיפים) ----------
CREATE TABLE adjustments (
    adjustment_id  INTEGER PRIMARY KEY,
    receipt_id     TEXT NOT NULL REFERENCES receipts (receipt_id),
    type           TEXT NOT NULL CHECK (type IN ('BANK_FEE', 'WITHHOLDING_TAX', 'FX_DIFFERENCE', 'ROUNDING')),  -- הגנה 2
    amount_minor   INTEGER NOT NULL CHECK (amount_minor <> 0),
    currency       TEXT NOT NULL CHECK (currency IN ('ILS', 'USD', 'EUR'))
) STRICT;


-- ---------- 6. יתרות פתוחות: תצוגה מחושבת, לא טבלה (הגנה 3) ----------
CREATE VIEW open_balances AS
SELECT
    i.invoice_id,
    i.customer_id,
    i.issue_date,
    i.currency,
    i.amount_minor,
    i.amount_minor - COALESCE(SUM(a.applied_minor), 0) AS open_minor,   -- יתרה פתוחה
    CASE
        WHEN i.amount_minor - COALESCE(SUM(a.applied_minor), 0) = 0              THEN 'PAID'
        WHEN i.amount_minor - COALESCE(SUM(a.applied_minor), 0) < i.amount_minor THEN 'PARTIALLY_PAID'
        ELSE 'OPEN'
    END AS status                                                          -- סטטוס מחושב
FROM invoices AS i
LEFT JOIN receipt_applications AS a ON a.invoice_id = i.invoice_id
GROUP BY i.invoice_id;


-- ---------- 7. כינויים (זיכרון הסוכן) ----------
CREATE TABLE aliases (
    bank_name        TEXT NOT NULL,
    customer_id      TEXT NOT NULL REFERENCES customers (customer_id),
    times_confirmed  INTEGER NOT NULL DEFAULT 0 CHECK (times_confirmed >= 0),
    PRIMARY KEY (bank_name, customer_id)       -- אותו שם יכול להופיע ללקוחות שונים (פייפלו)
) STRICT;


-- ---------- 8. תנועות שטופלו (זיכרון הסוכן) ----------
CREATE TABLE processed_bank_tx (
    tx_id         TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    result        TEXT NOT NULL CHECK (result IN ('POSTED', 'FILTERED', 'MANUAL')),  -- MANUAL = נדחתה, לטיפול ידני (ה-78)
    processed_at  TEXT NOT NULL,
    note          TEXT NOT NULL DEFAULT ''     -- למשל הסיבה לטיפול ידני
) STRICT;


-- ---------- 9. יומן ביקורת: כל החלטה של אדם ותוצאת הרישום (ה-78, רק מוסיפים) ----------
CREATE TABLE approval_audit (
    audit_id     INTEGER PRIMARY KEY,         -- מספר רץ אוטומטי
    run_id       TEXT NOT NULL,               -- באיזו ריצה
    tx_id        TEXT NOT NULL,               -- על איזו העברה
    proposal     TEXT NOT NULL,               -- ההצעה שהוצגה במסך (JSON): ודאות, לקוח, חשבוניות, סכומים
    action       TEXT NOT NULL CHECK (action IN ('APPROVE', 'REJECT', 'APPLY_TO_OLDEST')),  -- מה הוחלט
    approved_by  TEXT NOT NULL,               -- מי החליט
    decided_at   TEXT NOT NULL,               -- מתי (UTC)
    outcome      TEXT NOT NULL,               -- מה קרה ברישום: POSTED / MANUAL / CHANGED_SINCE_SHOWN ...
    receipt_id   TEXT,                        -- מספר התקבול, אם נרשם
    detail       TEXT NOT NULL DEFAULT ''     -- הסבר קצר, למשל למה הרישום נעצר
) STRICT;

CREATE INDEX idx_audit_run ON approval_audit (run_id);


-- =====================================================================
-- שמירה על הכללים בתוך המסד עצמו - גם אם יש באג בקוד
-- =====================================================================

-- החלה חייבת להיות במטבע של החשבונית, ולא יכולה להוריד את היתרה מתחת לאפס
-- או להעלות אותה מעל סכום החשבונית
CREATE TRIGGER applications_check
BEFORE INSERT ON receipt_applications
BEGIN
    SELECT RAISE(ABORT, 'application currency differs from invoice currency')
    WHERE NEW.currency <> (SELECT currency FROM invoices WHERE invoice_id = NEW.invoice_id);

    SELECT RAISE(ABORT, 'application exceeds open balance')
    WHERE NEW.applied_minor > (SELECT open_minor FROM open_balances WHERE invoice_id = NEW.invoice_id);

    SELECT RAISE(ABORT, 'reversal exceeds applied amount')
    WHERE (SELECT open_minor FROM open_balances WHERE invoice_id = NEW.invoice_id) - NEW.applied_minor
          > (SELECT amount_minor FROM invoices WHERE invoice_id = NEW.invoice_id);
END;

-- רשומות כספיות לא משתנות ולא נמחקות (ה-30). תיקון = רשומה הפוכה.
CREATE TRIGGER invoices_no_update BEFORE UPDATE ON invoices
BEGIN SELECT RAISE(ABORT, 'invoices are immutable'); END;
CREATE TRIGGER invoices_no_delete BEFORE DELETE ON invoices
BEGIN SELECT RAISE(ABORT, 'invoices are immutable'); END;

CREATE TRIGGER receipts_no_update BEFORE UPDATE ON receipts
BEGIN SELECT RAISE(ABORT, 'receipts are append-only'); END;
CREATE TRIGGER receipts_no_delete BEFORE DELETE ON receipts
BEGIN SELECT RAISE(ABORT, 'receipts are append-only'); END;

CREATE TRIGGER applications_no_update BEFORE UPDATE ON receipt_applications
BEGIN SELECT RAISE(ABORT, 'applications are append-only'); END;
CREATE TRIGGER applications_no_delete BEFORE DELETE ON receipt_applications
BEGIN SELECT RAISE(ABORT, 'applications are append-only'); END;

CREATE TRIGGER adjustments_no_update BEFORE UPDATE ON adjustments
BEGIN SELECT RAISE(ABORT, 'adjustments are append-only'); END;
CREATE TRIGGER adjustments_no_delete BEFORE DELETE ON adjustments
BEGIN SELECT RAISE(ABORT, 'adjustments are append-only'); END;

CREATE TRIGGER audit_no_update BEFORE UPDATE ON approval_audit
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON approval_audit
BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
