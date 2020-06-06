# Surface-Water-Quality-Data-Anomaly-Detection
Surface water quality data analysis and prediction of Potomac River, West Virginia, USA. Using time series forecasting, and anomaly detection : ARIMA, SARIMA, Isolation Forest, OCSVM and Gaussian Distribution

There exists an imperious need for development of schemes to analyse constantly monitored environmental data i.e. information about the various aspects of the ecosystem such as Surface Water Quality Parameters such as Dissolved Oxygen, Turbidity, Specific Conductance of water and analyse them for unnatural increase in their general values above predetermined standard levels to detect environmental anomalies that cause such increase. These parameters reflect the absolute state of the ecosystem of a particular geographical area, and thus help us to access any present or future discrepancies which can cause environmental degradation by direct or indirect activities of man in the geographical area. 

This process is done using Time Series forecasting techniques ARIMA and Seasonal ARIMA and anomaly detection techniques which are Isolation Forest, Gaussian Distribution, OneclassSVM, LSTM and Extended Isolation Forest.

<p>
  
  <h1> Working of Project:</h1>
  
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Formulae/Flowchart%20Major.png">
  
  
  
  
  <h3>Stationarity of Dataset</h3>
  <ol>
    <li>
      <h4>1. Augmented Dickey Fuller Test</h4>
      <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/ADF%20Test%20Turb.JPG">
      <h4>2. Rolling Mean Plot</h4>
      <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/Rolling%20Mean%20and%20STD%20Plot%20Turb.png">
    </li>
  </ol>
  
  
  
  
  
  <h3>Time Series Forecasting with : ARIMA</h3>
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Flowchart/arima%20flow.png">
  
  <h4> Result : ARIMA </h4>
   <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/ARIMA%20Original%20vs%20Predicted.png">
  
  <h4> Result : Seasonal ARIMA with window = 192(Daily number of observations)</h4>
   <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/Differenced%20ARIMA%20Original%20vs%20Predicted%202.png">

<h4>Time Series Forecast Result Analysis</h4>
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/Time%20Series%20Error%20Analysis.JPG">
  <hr>
  
  
  <h3>Isolation Forest Anomaly Detection</h3>
  <h4> iTree Generation and Anomaly Score Calculation<h4>
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Flowchart/Isolation%20Tree%20working.jpeg">
  
   <h4>Isolation Forest</h4>
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Flowchart/iForest.jpeg">             
  <h4> Result : iForest Anomaly Detection </h4>                                                          
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/iForest%20Anomaly%20Detection%20New.png">                                   
  <hr>
   
   
   
   <h3>OneClassSVM</h3>
    <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Flowchart/OCSVM%202.jpeg">
    <h4> Result : OneClassSVM </h4>         
    <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/OCSVM%20Anomaly%20Detection.png">   
    <hr>
    
    
    
  <h3>Gaussian Distribution</h3>
    <h4> Result </h4>         
    <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/Gaussian%20Anomaly%20Detection.png">   
    <hr>
    
    
    
  <h3>Anomaly Detection Result Analysis</h3>
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/Anomalies%20Detected%20Analysis%20New.JPG">
   The above graph shows that isolation forest may be detecting a lot more false positives than the other approaches or it might be over measuring the result. All other methods give similar result with anomaly percentage ranging from 9 to 20 %. The Anomaly graph predictions shown earlier indicate that most anomalies occur on 29 January, 2017 and also on 22 March, 2017. These anomalies can be acknowledged by the fact that these dates had actually shown intensity rainfalls on the monitoring site.

  
 </p>
