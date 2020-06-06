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
  <h2> Result : </h2>
   <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Results/1/ARIMA%20Original%20vs%20Predicted.png">

  <hr>
  <h3>Isolation Forest</h3>
  <h2> iTree Generation and Anomaly Score Calculation<h2>
  <img src=https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Flowchart/Isolation%20Tree%20working.jpeg">
   <h3>Isolation Forest</h3>
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Flowchart/iForest.jpeg">                                                                                                                                            
  <hr>
   
   <h3>OneClassSVM</h3>
  <img src="https://github.com/absaw/Surface-Water-Quality-Data-Anomaly-Detection/blob/master/Images/Flowchart/OCSVM%202.jpeg">
    <hr>

  
    <img src="">

  
 </p>
