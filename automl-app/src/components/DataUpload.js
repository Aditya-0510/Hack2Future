import React, { useState } from 'react';
import axios from 'axios';

const DataUpload = () => {
    const [file, setFile] = useState(null);
    const [dataInfo, setDataInfo] = useState(null);
    const [trainingResult, setTrainingResult] = useState(null);
    const [algorithm, setAlgorithm] = useState('random_forest');
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append('file', file);
        setLoading(true);
    
        try {
            const response = await axios.post('http://127.0.0.1:8000/upload-data/', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setDataInfo(response.data.data_info);
            setError(null); // Clear any previous errors
            setFile(null); // Reset file state
        } catch (error) {
            console.error('Error uploading file:', error);
            if (error.response) {
                // Server responded with a status other than 200 range
                setError(`Error: ${error.response.data.detail || error.response.statusText}`);
            } else {
                // Network error or other errors
                setError('Network error. Please try again.');
            }
        } finally {
            setLoading(false); // End loading
        }
    };
    

    const handleTrainModel = async () => {
        const formData = new FormData();
        formData.append('file', file); // Ensure 'file' is appended correctly
        formData.append('algorithm', algorithm); // Append the chosen algorithm
    
        try {
            const response = await axios.post('http://127.0.0.1:8000/train-model/', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setTrainingResult(response.data.accuracy); // Capture model accuracy
        } catch (error) {
            console.error('Error training model:', error);
            if (error.response) {
                alert(`Error: ${error.response.data.detail || error.response.statusText}`);
            } else if (error.request) {
                alert('Error: No response received from the server.');
            } else {
                alert(`Error: ${error.message}`);
            }
        }
    };
    
    
    

    return (
        <div>
            <h2>Upload Dataset</h2>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload} disabled={loading || !file}>Upload</button>

            {loading && <p>Loading...</p>}
            {error && <p style={{ color: 'red' }}>{error}</p>}

            {dataInfo && (
                <div>
                    <h3>Data Preview:</h3>
                    <p>Columns: {dataInfo.columns.join(', ')}</p>
                    <p>Shape: {dataInfo.shape[0]} rows, {dataInfo.shape[1]} columns</p>
                    <pre>{JSON.stringify(dataInfo.preview, null, 2)}</pre>
                </div>
            )}

            {dataInfo && (
                <div>
                    <h3>Select Algorithm</h3>
                    <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
                        <option value="random_forest">Random Forest</option>
                        <option value="logistic_regression">Logistic Regression</option>
                        <option value="decision_tree">Decision Tree</option>
                    </select>
                    <button onClick={handleTrainModel} disabled={!dataInfo}>Train Model</button>
                </div>
            )}

            {trainingResult && (
                <div>
                    <h3>Model Training Result:</h3>
                    <p>Accuracy: {trainingResult}</p>
                </div>
            )}
        </div>
    );
};

export default DataUpload;
