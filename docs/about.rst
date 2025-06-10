About DILImap
-------------

**Why DILI Remains a Critical Challenge in Drug Development**

Drug-Induced Liver Injury (DILI) is a one of the most significant barriers in drug development,
responsible for costly late-stage failures and market withdrawals. Why do existing preclinical
models struggle to adequately predict DILI? DILI arises from complex, multifactorial mechanisms,
including mitochondrial dysfunction, oxidative stress, and reactive metabolite formation, with
idiosyncratic cases being especially unpredictable. Current preclinical methods, such as QSAR
models and in vitro assays, offer limited sensitivity and and fail to capture the underlying
complexity of DILI. Despite advances like 3D assays, these approaches remain reductionist and
fail to address the underlying causes of DILI. This underscores the urgent need for more
comprehensive and predictive models to mitigate late-stage failures and improve drug safety.

**Introducing DILImap: Building a Foundation for Toxicogenomics**

Toxicogenomics offers a transformative approach to understanding and predicting DILI. By
leveraging gene expression signatures indicative of DILI, this method can predict DILI risk and
provide insights into the molecular mechanisms of DILI. We believe this method is a potential
game-changer as it provides a distinct edge in its predictive performance,
comprehensive mechanistic coverage and scalability. **DILImap** is a comprehensive RNA-seq library
featuring 300 compounds tested across various concentrations, built to capture the complexity of
DILI mechanisms. It leverages resources like DILIrank and LiverTox to categorize compounds based
on their liver injury history. This dataset provides the foundation for uncovering molecular
pathways and early gene signatures indicative of liver injury.

**Introducing ToxPredictor: A Machine Learning Model for DILI Prediction**

ToxPredictor, a machine learning model trained on DILImap, achieves 88% sensitivity in
detecting  DILI-positive compounds and 100% specificity in identifying safe compounds in
blind validation. This performance surpasses 20+ preclinical methods. ToxPredictor uses a random
forest algorithm to estimate dose-specific DILI risk probabilities. The model is trained on
pathway-level signatures derived from differentially expressed genes identified relative to DMSO
controls. Enrichment analysis, performed using WikiPathways, pinpoints biological pathways
disrupted by DILI-associated compounds, providing a deeper understanding of the molecular
mechanisms at play. By modeling probabilities across different concentrations, ToxPredictor
estimates a transcriptomic first DILI dose, defined as the first dose where the predicted
probability exceeds 0.7. Safety margins are quantified as the ratio of the transcriptomic first
DILI dose to the maximum plasma concentration (Cmax) at therapeutic levels. A safety margin
threshold of 80 was established as the optimal cutoff to classify compounds as high/low risk.