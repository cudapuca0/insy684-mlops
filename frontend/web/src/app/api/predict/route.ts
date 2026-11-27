export async function POST(request: Request) {
  const apiBaseUrl = process.env.API_BASE_URL;

  if (!apiBaseUrl) {
    return Response.json({ error: "Prediction API is not configured." }, {
      status: 503,
    });
  }

  try {
    const prediction = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: await request.text(),
      signal: AbortSignal.timeout(10_000),
    });

    return new Response(await prediction.text(), {
      status: prediction.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json({ error: "Could not reach the prediction API." }, {
      status: 502,
    });
  }
}
