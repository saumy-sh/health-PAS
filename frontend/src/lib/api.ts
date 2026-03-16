export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export async function uploadFiles(files: File[]) {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  console.log(`[API] Uploading ${files.length} files...`);
  const res = await fetch(`${API_BASE}/analyse/`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.text();
    console.error(`[API] Upload failed: ${err}`);
    throw new Error("Upload failed");
  }
  const data = await res.json();
  console.log(`[API] Upload success. Session: ${data.session_id}`);
  return data;
}

export async function runAgent1(filePaths: string[]) {
  console.log(`[API] Running Agent 1 on ${filePaths.length} files...`);
  const res = await fetch(`${API_BASE}/analyse/document_ocr`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(filePaths),
  });
  if (!res.ok) {
    console.error(`[API] Agent 1 failed: ${res.statusText}`);
    throw new Error("Agent 1 failed");
  }
  const data = await res.json();
  console.log(`[API] Agent 1 complete. Identified ${data.documents?.length} docs.`);
  return data;
}

export async function runAgent2(agent1Result: any) {
  console.log(`[API] Running Agent 2...`);
  const res = await fetch(`${API_BASE}/analyse/policy_checker`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agent1Result),
  });
  if (!res.ok) {
    console.error(`[API] Agent 2 failed: ${res.statusText}`);
    throw new Error("Agent 2 failed");
  }
  const data = await res.json();
  console.log(`[API] Agent 2 complete. Policy: ${data.insurer_name}`);
  return data;
}

export async function runAgent3(agent2Result: any) {
  console.log(`[API] Running Agent 3...`);
  const res = await fetch(`${API_BASE}/analyse/policy_retriever`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agent2Result),
  });
  if (!res.ok) {
    console.error(`[API] Agent 3 failed: ${res.statusText}`);
    throw new Error("Agent 3 failed");
  }
  const data = await res.json();
  console.log(`[API] Agent 3 complete. Req count: ${data.document_requirements?.length}`);
  return data;
}

export async function runAgent4(agent3Result: any) {
  console.log(`[API] Running Agent 4...`);
  const res = await fetch(`${API_BASE}/analyse/document_checker`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agent3Result),
  });
  if (!res.ok) {
    console.error(`[API] Agent 4 failed: ${res.statusText}`);
    throw new Error("Agent 4 failed");
  }
  const data = await res.json();
  console.log(`[API] Agent 4 complete. Can proceed: ${data.can_proceed}`);
  return data;
}

export async function runAgent5(agent4Result: any) {
  console.log(`[API] Running Agent 5...`);
  const res = await fetch(`${API_BASE}/analyse/eligibility_reasoning`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agent4Result),
  });
  if (!res.ok) {
    console.error(`[API] Agent 5 failed: ${res.statusText}`);
    throw new Error("Agent 5 failed");
  }
  const data = await res.json();
  console.log(`[API] Agent 5 complete. Determination: ${data.determination}`);
  return data;
}

export async function runAgent6(agent5Result: any) {
  console.log(`[API] Running Agent 6...`);
  const res = await fetch(`${API_BASE}/analyse/form_filler`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(agent5Result),
  });
  if (!res.ok) {
    console.error(`[API] Agent 6 failed: ${res.statusText}`);
    throw new Error("Agent 6 failed");
  }
  const data = await res.json();
  console.log(`[API] Agent 6 complete. PDF: ${data.filled_pdf_path}`);
  return data;
}
