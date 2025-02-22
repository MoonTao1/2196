# Driving Fixation Prediction via Unsupervised Domain Adaptation for Sunny-to-Rainy Traffic Scenes

​	[note] We will release our complete code after the paper is **accepted** ✔️! Please look forward to it.🕓

## 📰 News

**[2024.09.15]** 🎈Everyone can download the ADED dataset, the data is stored on BaiduNetdisk, from 

**[2024.010.17]** 🎈We propose $VP^2Net$, which is a visual dual perception-inspired network for exploring the causes of driver’s attention shifts.

**[2024.12.16]** 🎈We will submit the article to ***ICME*** (IEEE International Conference on Multimedia and Expo ).😃

## ✨Model

<img src="pic\model3.png" style="zoom:70%;" />

>The architecture of the proposed model. The model consists of three structures, two feature extractors with shared parameters, a predictor and a domain adaptation module. The target domain data does not pass through the predictor, and only the source domain data passes through the predictor to calculate the loss value. The data of the two domains need to go through the domain classifier for domain discrimination, and a classification loss is obtained.

## 💻Dataset

<div align="center">
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Train (video/frame)</th>
      <th>Valid (video/frame)</th>
      <th>Test (video/frame)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TrafficGaze</td>
      <td>49080</td>
      <td>6655</td>
      <td>19135</td>
    </tr>
    <tr>
      <td>DrFixD-rainy</td>
      <td>52291</td>
      <td>9816</td>
      <td>19154</td>
    </tr>
  </tbody>
</table>
</div>
<img src="pic\12.png" style="zoom:100%;" />

## 🚀 Quantitative Analysis
>COMPARISON WITH OTHER METHODS FROM TraffiicGaze TO DRFIXD(RAINY)

| Model      | AUC B↑     | AUC J↑     | NSS↑       | CC↑        | SIM↑       | KLD↓       |
| ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Itti       | 0.8426     | 0.8570     | 1.5569     | 0.3695     | 0.2702     | 1.6868     |
| ImageSig   | 0.6368     | 0.6912     | 0.5298     | 0.1404     | 0.1781     | 2.4159     |
| HFT        | 0.7670     | 0.7867     | 1.0077     | 0.2311     | 0.2240     | 2.0079     |
| DDC        | 0.7359     | 0.7608     | 0.9433     | 0.2492     | 0.2454     | 1.8950     |
| Deep Coral | 0.7132     | 0.7680     | 1.1630     | 0.3135     | 0.2909     | 1.8543     |
| DRCN       | 0.6021     | 0.7676     | 0.8007     | 0.2307     | 0.2688     | 2.5432     |
| AT         | 0.7498     | 0.7440     | 0.9608     | 0.2578     | 0.2634     | 1.7583     |
| CPFE       | 0.8147     | 0.9201     | 2.5645     | 0.5720     | 0.4722     | 1.0185     |
| FBL        | 0.8172     | **0.9395** | **3.7492** | 0.7279     | 0.5716     | 0.7778     |
| Ours       | **0.8612** | 0.9320     | 3.3928     | **0.7442** | **0.5965** | **0.7510** |

>COMPARISION OF FPS AND PARAMETERS


| Model      | Parameter | FPS        |
| ---------- | --------- | ---------- |
| DDC        | **9.90M** | 82.91      |
| Deep Coral | 378.17M   | 35.85      |
| DRCN       | 191.70M   | 95.14      |
| AT         | 11.20M    | 170.78     |
| CPFE       | 16.38M    | 46.76      |
| FBL        | 87.48M    | 33.19      |
| Ours       | **9.90M** | **173.79** |

## 🚀Visualisation of intermediate results
>Qualitative evaluation comparison of proposed model and the other methods from sunny dataset TrafficGaze to rainy dataset DrFixD(rainy). The circles highlight objects/areas in the driving scene that disrupt the driver's attention.
<img src="pic\AM.png" style="zoom:80%;" />


>Visualization results of rainy weather forecast under complex scenarios, yellow circles indicate unrelated factors that may capture the driver's attention, including (a) The wiper blocks the line of sight, (b) Reflection from stagnant water, (c) Glare in the taillights of vehicles ahead in dim light.


<img src="pic\11.png" style="zoom:60%;" />




## 📝Model Zoo

We give the weights obtained by training in the paper. Includes weights from ablation experiments. These weights may be able to be used as your pre-training weights, reducing the time required for learning.

| Model                | pth           | Model                     | pth           |
| -------------------- | ------------- | ------------------------- | ------------- |
| $VP^2Net$            | [BaiduYun][2] | $VP^2Net$ w/ Add          | [BaiduYun][6] |
| $VP^2Net$ w/ offline | [BaiduYun][3] | $VP^2Net$ w/ Add-Multiply | [BaiduYun][7] |
| $VP^2Net$ w/ online  | [BaiduYun][2] | $VP^2Net$ w/ DER-Net      | [BaiduYun][8] |
| $VP^2Net$ w/o APE    | [BaiduYun][5] | $VP^2Net$ w/ (2D)         | [BaiduYun][9] |
| $VP^2Net$ w/o PEM    | [BaiduYun][5] | $VP^2Net$ w/ (3D)         | [BaiduYun][2] |

[2]: https://pan.baidu.com/s/1YgmhD9Nq8AAkEKrXYsMTDA?pwd=V2PN "V2PNet"
[3]: https://pan.baidu.com/s/1WdVunAkihHX9DZPGDga38Q?pwd=V2PN "offline"
[5]: https://pan.baidu.com/s/1lG9Cn7l8TjcA9C28Ukq7xQ?pwd=V2PN "w/o pem"
[6]: https://pan.baidu.com/s/18TFOhjXw-FqdNFkaLjjLDQ?pwd=V2PN "w/ add"
[7]: https://pan.baidu.com/s/1OD_xuD2X0OOgGNk13RnEog?pwd=V2PN "w/ add-multiply"
[8]:https://pan.baidu.com/s/1Wb_mTrpTx0A5LqPLXvRenA?pwd=V2PN "DER-Net"
[9]: https://pan.baidu.com/s/1QaHok0aCX94tcCAKUldm3Q?pwd=V2PN "w/ 2d"



## 💖Support the Project

Thanks to the open-source video action detection models (ViViT, VideoMAE) at [huggingface🤗][10]  for supporting this paper.

[10]: https://huggingface.curated.co/	"huggingface"

## 📄Cite

Welcome to our work ! 

