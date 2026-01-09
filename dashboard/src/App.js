import { useState } from "react";

function App() {
  const [form, setForm] = useState({
    cost: "",
    stock: "",
    day: "",
    week: "",
    month: "",
    is_weekend: 0,
    competitor: "",
    discount: "",
    holiday: 0,
    last_week_sales: "",
    category: ""
  });

  const [price, setPrice] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const predict = async () => {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form)
    });

    const data = await res.json();
    setPrice(data.recommended_price);
  };

  return (
    <div style={{ padding: 30 }}>
      <h2>📊 PriceOptima Dashboard</h2>

      {Object.keys(form).map((key) => (
        <div key={key} style={{ marginBottom: 10 }}>
          <input
            name={key}
            placeholder={key.replace("_", " ")}
            onChange={handleChange}
            style={{ width: 200 }}
          />
        </div>
      ))}

      <button onClick={predict}>Predict Price</button>

      <h3>Recommended Price: {price}</h3>
    </div>
  );
}

export default App;
