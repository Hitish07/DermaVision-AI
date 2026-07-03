# API Reference

## `POST /api/scan`
**Payload**: `multipart/form-data` (file)
**Returns**: `ScanResultResponse`
Validates, predicts, and initializes the scan cycle.

## `GET /api/explainability/{scan_id}`
**Returns**: `AsyncXAIResponse`
Used by the frontend to poll for SHAP and Integrated Gradients completion.

## `GET /api/history`
**Returns**: `List[HistoryItem]`
Retrieves the paginated ledger of historical inferences.
