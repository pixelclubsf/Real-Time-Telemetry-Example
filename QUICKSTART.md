# Solar Boat Dashboard - Quick Start Guide

Your Flask-based interactive dashboard is ready to use!

## ✅ Installation Complete

All dependencies have been installed successfully:
- Flask 3.0.0
- Plotly 5.17.0 (interactive charts)
- Matplotlib 3.8.0
- NumPy 1.24.3
- MGRS 1.4.6
- PROJ (installed via Homebrew)

## 🚀 How to Run

1. **Navigate to the project directory:**
   ```bash
   cd /Users/charlie/Desktop/Solar-Regatta
   ```

2. **Start the Flask server:**
   ```bash
   python app.py
   ```

3. **Open in your browser:**
   ```
   http://localhost:5001
   ```

## 📊 Using the Dashboard

1. **Load Sample Data**
   - Click the "📊 Load Sample Data" button
   - Optionally adjust Duration and Interval
   - Wait for the dashboard to populate

2. **View Your Data**
   - **Metrics Cards**: Summary statistics at the top
   - **Speed vs Time**: Main performance graph
   - **Battery Voltage**: Power system monitoring
   - **Motor Current**: Energy consumption tracking
   - **Efficiency Plot**: Speed vs current relationship
   - **GPS Track**: Position data visualization

3. **Interact with Charts**
   - **Hover**: See exact values
   - **Zoom**: Click and drag
   - **Pan**: Shift + drag
   - **Reset**: Double-click

4. **Export Data**
   - Click "💾 Export Data" to download JSON file
   - Contains all metrics and raw data points

## 📁 Project Structure

```
Solar-Regatta/
├── app.py                      # Flask server
├── solar.py                    # Core analysis functions
├── requirements.txt            # Python packages
├── FLASK_README.md            # Full documentation
├── QUICKSTART.md              # This file
├── templates/
│   └── dashboard.html         # Main UI
└── static/
    ├── css/
    │   └── style.css          # Styling
    └── js/
        └── dashboard.js       # Interactivity
```

## 🔧 Customization

### Change Port
Edit `app.py` line 315:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5001 to your port
```

### Add Your Own Data
Replace `generate_sample_vesc_data()` call in `app.py` with your data source

### Modify Chart Colors
Edit colors in `app.py` functions:
- Speed: `#2E86AB`
- Voltage: `#A23B72`
- Current: `#F18F01`

## 🐛 Troubleshooting

**Port Already in Use**
```bash
# Find what's using port 5001
lsof -i :5001

# Kill the process
kill -9 <PID>

# Or use a different port in app.py
```

**Module Not Found**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Charts Not Loading**
- Check browser console (F12)
- Ensure internet connection (Plotly CDN needed)
- Clear browser cache

## 📈 Next Steps

1. **Load Your VESC Data**: Modify the data loading function to read your actual VESC logs
2. **Add More Charts**: Edit `app.py` to create additional visualizations
3. **Database Integration**: Store historical race data in a database
4. **Real-time Updates**: Implement WebSockets for live race monitoring

## 🎯 Features Included

✅ Interactive speed vs time graph
✅ Battery voltage monitoring
✅ Motor current tracking
✅ Efficiency analysis
✅ GPS track visualization
✅ Performance metrics
✅ Data export (JSON)
✅ Responsive design
✅ Sample data generation
✅ Error handling

## 📞 Support

Check the full documentation in `FLASK_README.md` for:
- API endpoints
- Advanced customization
- Database setup
- Deployment options

Enjoy your solar boat analysis dashboard! ☀️
