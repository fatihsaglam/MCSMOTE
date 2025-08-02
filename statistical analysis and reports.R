library(data.table)
library(readxl)
library(PMCMRplus)
library(tidyr)

data <- read_excel(path = "Results_Resampling_Experiments.xlsx")
data <- as.data.frame(data)
data

# Metriğe karşılık gelen sütunları bul
metric_cols <- c("BACC", "MCC", "GMEAN", "F1", "AUC", "PRAUC", "Runtime")

# Veriyi uzun formata çevir (her metrik ayrı satıra dönüşür)
df_long <- data %>%
  pivot_longer(cols = all_of(metric_cols), 
               names_to = "Metric",
               values_to = "Value")

# Her Resampler için ayrı sütun olacak şekilde genişlet
df_wide <- df_long %>%
  select(-BestHyperparResampler) %>%  # bu kolonu istersen tutabilirsin
  pivot_wider(names_from = Resampler, values_from = Value)

# Sonuç
head(df_wide)

data$BestHyperparResampler[data$Resampler == "MCSMOTE"]


f_par2data <- function(textVector) {
  do.call(rbind, lapply(textVector, function(x) {
    kv <- strsplit(x, ";")[[1]]
    vals <- setNames(
      as.numeric(sub(".*=", "", kv)),
      sub("=.*", "", kv)
    )
    as.data.frame(as.list(vals))
  }))
}

