import { useState } from "react";

function App() {
  // Form state
  const [formData, setFormData] = useState({
    area: "",
    rooms: "",
    floor: "",
    free_from: "unknown",
    has_balkon: false,
    has_terrasse: false,
    has_garten: false,
    elevator: false,
    parking: false,
    has_basement: false,
    is_barrier_free: false,
    has_built_in_kitchen: false,
    has_bathtub: false,
    has_shower: false,
    flooring_type: "unknown",
    energy_source: "Other",
    heating_type: "Unknown",
    property_condition: "Unknown",
    year_built: "",
    district: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const payload = {
        area: parseFloat(formData.area),
        rooms: parseInt(formData.rooms),
        floor: parseInt(formData.floor),
        free_from: formData.free_from,
        has_balkon: formData.has_balkon,
        has_terrasse: formData.has_terrasse,
        has_garten: formData.has_garten,
        elevator: formData.elevator,
        parking: formData.parking,
        has_basement: formData.has_basement,
        is_barrier_free: formData.is_barrier_free,
        has_built_in_kitchen: formData.has_built_in_kitchen,
        has_bathtub: formData.has_bathtub,
        has_shower: formData.has_shower,
        flooring_type: formData.flooring_type,
        energy_source: formData.energy_source,
        heating_type: formData.heating_type,
        property_condition: formData.property_condition,
        year_built: parseInt(formData.year_built),
        district: formData.district,
      };
      
      const response = await fetch("https://model-hosting-2u0x.onrender.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      console.log("Backend response:", data);
      console.log("Prediction value:", data.prediction);

      if (response.ok) {
        setPrediction(data.prediction);
      } else {
        setError(data.error || "Prediction failed");
      }
    } catch (err) {
      setError(
        "Failed to connect to backend. Make sure the backend is running.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        padding: "0px 60px 5px 60px",
        margin: "0 auto",
        backgroundImage:
          "linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.2)), url('/berlin.jpg')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        minHeight: "99.3vh",
        filter: "contrast(1.1) brightness(0.9)",
        color: "white",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start",
      }}
    >
      <h1 className="text-center w-full text-4xl font-bold">
        Rent Prediction in Berlin
      </h1>

      <form onSubmit={handleSubmit} style={{ display: "grid", gap: "5px" }}>
        {/* Basic Information */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "15px",
          }}
        >
          <div>
            <label>Area (sqm)*</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="10000"
              name="area"
              value={formData.area}
              onChange={handleInputChange}
              required
              style={{ width: "100%", padding: "8px 0px" }}
            />
            <small style={{ color: "#d0cece", fontWeight: "bold" }}>
              Range: 0 - 10,000 sqm
            </small>
          </div>
          <div>
            <label>Rooms*</label>
            <select
              name="rooms"
              value={formData.rooms}
              onChange={handleInputChange}
              required
              style={{ width: "100%", padding: "8px" }}
            >
              <option value="">Select rooms</option>
              {[...Array(20)].map((_, i) => {
                const base = Math.floor(i / 2) + 1;
                const isHalf = i % 2 === 1;
                const value = isHalf ? `${base}.5` : base.toString();
                return (
                  <option key={value} value={value}>
                    {value}
                  </option>
                );
              })}
            </select>
            <small style={{ color: "#d0cece" }}>Range: 1 - 10 rooms</small>
          </div>
          <div>
            <label>Floor*</label>
            <select
              name="floor"
              value={formData.floor}
              onChange={handleInputChange}
              required
              style={{ width: "100%", padding: "8px" }}
            >
              <option value="">Select floor</option>
              <option value="-1">Basement (-1)</option>
              <option value="0">Ground Floor (0)</option>
              {[...Array(11)].map((_, i) => (
                <option key={i + 1} value={i + 1}>
                  {i + 1}. Floor
                </option>
              ))}
            </select>
            <small style={{ color: "#d0cece" }}>
              Range: Basement (-1) to 11th floor
            </small>
          </div>
        </div>

        {/* Year Built and District */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "15px",
          }}
        >
          <div>
            <label>Year Built</label>
            <input
              type="number"
              name="year_built"
              value={formData.year_built}
              onChange={handleInputChange}
              min="1700"
              max={new Date().getFullYear()}
              style={{ width: "100%", padding: "8px 0px" }}
            />
            <small style={{ color: "#d0cece" }}>
              Range: 1700 - {new Date().getFullYear()}
            </small>
          </div>
          <div>
            <label>District*</label>
            <select
              name="district"
              value={formData.district}
              onChange={handleInputChange}
              style={{ width: "100%", padding: "8px" }}
            >
              <option value="">Select district</option>
              <option value="Alt-Treptow">Alt-Treptow</option>
              <option value="Altglienicke">Altglienicke</option>
              <option value="Blankenburg">Blankenburg</option>
              <option value="Bohnsdorf">Bohnsdorf</option>
              <option value="Britz">Britz</option>
              <option value="Buch">Buch</option>
              <option value="Charlottenburg">Charlottenburg</option>
              <option value="Dahlem">Dahlem</option>
              <option value="Friedenau">Friedenau</option>
              <option value="Französisch Buchholz">Französisch Buchholz</option>
              <option value="Friedrichshain">Friedrichshain</option>
              <option value="Friedrichsfelde">Friedrichsfelde</option>
              <option value="Friedrichshagen">Friedrichshagen</option>
              <option value="Gesundbrunnen">Gesundbrunnen</option>
              <option value="Grunewald">Grunewald</option>
              <option value="Grünau">Grünau</option>
              <option value="Hellersdorf">Hellersdorf</option>
              <option value="Hermsdorf">Hermsdorf</option>
              <option value="Hohenschönhausen">Hohenschönhausen</option>
              <option value="Johannisthal">Johannisthal</option>
              <option value="Kladow">Kladow</option>
              <option value="Köpenick">Köpenick</option>
              <option value="Kreuzberg">Kreuzberg</option>
              <option value="Lichtenberg">Lichtenberg</option>
              <option value="Lichtenrade">Lichtenrade</option>
              <option value="Lichterfelde">Lichterfelde</option>
              <option value="Mahlsdorf">Mahlsdorf</option>
              <option value="Marienfelde">Marienfelde</option>
              <option value="Marzahn">Marzahn</option>
              <option value="Mitte">Mitte</option>
              <option value="Neukölln">Neukölln</option>
              <option value="Niederschönhausen">Niederschönhausen</option>
              <option value="Niederschöneweide">Niederschöneweide</option>
              <option value="Nikolassee">Nikolassee</option>
              <option value="Oberschöneweide">Oberschöneweide</option>
              <option value="Pankow">Pankow</option>
              <option value="Prenzlauer Berg">Prenzlauer Berg</option>
              <option value="Rahnsdorf">Rahnsdorf</option>
              <option value="Reinickendorf">Reinickendorf</option>
              <option value="Schöneberg">Schöneberg</option>
              <option value="Siemensstadt">Siemensstadt</option>
              <option value="Spandau">Spandau</option>
              <option value="Steglitz">Steglitz</option>
              <option value="Staaken">Staaken</option>
              <option value="Tempelhof">Tempelhof</option>
              <option value="Tegel">Tegel</option>
              <option value="Tiergarten">Tiergarten</option>
              <option value="Wannsee">Wannsee</option>
              <option value="Wedding">Wedding</option>
              <option value="Weißensee">Weißensee</option>
              <option value="Wilmersdorf">Wilmersdorf</option>
              <option value="Wittenau">Wittenau</option>
              <option value="Zehlendorf">Zehlendorf</option>
            </select>
          </div>
          <div>
            <label>Available From</label>
            <select
              name="free_from"
              value={formData.free_from}
              onChange={handleInputChange}
              style={{ width: "100%", padding: "8px" }}
            >
              <option value="immediately">Immediately</option>
              <option value="not_immediately">Not Immediately</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>
        </div>

        {/* Property Details - Selects */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "15px",
          }}
        >
          <div>
            <label>Flooring Type</label>
            <select
              name="flooring_type"
              value={formData.flooring_type}
              onChange={handleInputChange}
              style={{ width: "100%", padding: "8px" }}
            >
              <option value="unknown">Unknown</option>
              <option value="Wood">Wood</option>
              <option value="Laminate">Laminate</option>
              <option value="Tiles">Tiles</option>
              <option value="Vinyl">Vinyl</option>
              <option value="Stone">Stone</option>
              <option value="Carpet">Carpet</option>
            </select>
          </div>
          <div>
            <label>Energy Source</label>
            <select
              name="energy_source"
              value={formData.energy_source}
              onChange={handleInputChange}
              style={{ width: "100%", padding: "8px" }}
            >
              <option value="District Heating">District Heating</option>
              <option value="Gas">Gas</option>
              <option value="Hybrid">Hybrid</option>
              <option value="Heat Pump / Geothermal">
                Heat Pump / Geothermal
              </option>
              <option value="Oil/Coal">Oil / Coal</option>
              <option value="Electric">Electric</option>
              <option value="Other">Other</option>
            </select>
          </div>
          <div>
            <label>Heating Type</label>
            <select
              name="heating_type"
              value={formData.heating_type}
              onChange={handleInputChange}
              style={{ width: "100%", padding: "8px" }}
            >
              <option value="Central Heating">Central Heating</option>
              <option value="Underfloor Heating">Underfloor Heating</option>
              <option value="Unit Heating">Unit Heating</option>
              <option value="Individual">Individual</option>
              <option value="Unknown">Unknown</option>
            </select>
          </div>
        </div>

        {/* Property Condition */}
        <div>
          <label>Property Condition</label>
          <select
            name="property_condition"
            value={formData.property_condition}
            onChange={handleInputChange}
            style={{ width: "100%", padding: "8px" }}
          >
            <option value="New / First Occupancy">New / First Occupancy</option>
            <option value="Well Maintained">Well Maintained</option>
            <option value="Like New">Like New</option>
            <option value="Renovated">Renovated</option>
            <option value="Old Building">Old Building</option>
            <option value="Needs Renovation">Needs Renovation</option>
            <option value="Partially Renovated">Partially Renovated</option>
            <option value="Other">Other</option>
            <option value="Unknown">Unknown</option>
          </select>
        </div>

        {/* Property Features - Checkboxes (bottom) */}
        <div>
          <h3>Property Features</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
              gap: "5px",
            }}
          >
            {[
              { name: "has_balkon", label: "Balcony" },
              { name: "has_terrasse", label: "Terrace" },
              { name: "has_garten", label: "Garden" },
              { name: "elevator", label: "Elevator" },
              { name: "parking", label: "Parking" },
              { name: "has_basement", label: "Basement" },
              { name: "is_barrier_free", label: "Barrier Free" },
              { name: "has_built_in_kitchen", label: "Built-in Kitchen" },
              { name: "has_bathtub", label: "Bathtub" },
              { name: "has_shower", label: "Shower" },
            ].map(({ name, label }) => (
              <label
                key={name}
                style={{ display: "flex", alignItems: "center", gap: "5px" }}
              >
                <input
                  type="checkbox"
                  name={name}
                  checked={formData[name]}
                  onChange={handleInputChange}
                />
                {label}
              </label>
            ))}
          </div>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginTop: "10px",
          }}
        >
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "16px 50px",
              backgroundColor: loading ? "#d0cece" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: "8px",
              maxWidth: "350px",
              fontSize: "18px",
              fontWeight: "bold",
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Predicting..." : "Predict Rent"}
          </button>
        </div>
      </form>

      {error && (
        <div
          style={{
            color: "red",
            marginTop: "5px",
            padding: "10px",
            border: "1px solid red",
            borderRadius: "4px",
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}

      {prediction !== null && prediction !== undefined && (
        <div
          style={{
            marginTop: "0px",
            padding: "10px 20px 20px 20px",
            borderRadius: "4px",
          }}
        >
          <h2>
            Predicted Rent:{" "}
            {typeof prediction === "number" ? prediction.toFixed(2) : "N/A"} €
          </h2>
          {typeof prediction === "number" && (
            <>
              <small style={{ display: "block", marginTop: "5px", fontWeight:700, color: "#ffffff" }}>
                Real value may vary: {(prediction * 0.89).toFixed(2)} € - {(prediction * 1.11).toFixed(2)} €
              </small>
              <small style={{ display: "block", marginTop: "2px", color: "#ffffff" }}>
                Paying more? You might be overpaying. Paying less? You might be underpaying.
              </small>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
