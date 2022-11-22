# Date Created: Nov 18, 2022
#
# Use NN Model Tracker Excel file to make a dot plot of model performance vs run time
# 

library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(cowplot)
source("clean_ggplot.R")


DOT_IN_FILENAME = "../model_output/NN_Model_Results_Tracking_11-18-2022.xlsx"
LINE_IN_FILENAME = "../model_output/NN_Model_Results_Tracking_Line_11-18-2022.xlsx"

# PYTORCH_COLOR <- "#FFF2CC"
VIZ_COLORS <- c("PyT" = "chocolate1",
                "TF" = "deepskyblue1")
# VIZ_COLORS <- c("PyT" = "#FFE699", # pale yellow
#                 "TF" = "#BDD7EE") # pale blue
GEORGE_MASON_COLOR <- "#116020"
# PYTORCH_COLOR <- "#FFE699"
# TENSORFLOW_COLOR <- "#BDD7EE"

SELECTED_TRIALS <- c(33, 47, 50, 51, 52, 53, 55, 56, 57, 58)

read_data <- function(trial_filter, in_filename) {
    data <- read_excel(in_filename) %>%
        filter(Trial %in% trial_filter) %>%
        mutate(`Run Date` = Date,
               Packages = factor(if_else(Packages == "PyTorch", "PyT", "TF")),
               Momentum = `Momentum Rate`,
               LossFunction = factor(if_else(`Loss Function` == "NLLLoss", "NLL",
                                    if_else(`Loss Function` == "categorical crossentropy", "CC", "SCC"))),
               Layers = factor(Layers),
               LearnRate = `Learning\r\nRate`,
               TestingAccuracy = as.double(`Testing\r\nAccuracy`),
               `% Cases with\n100% Testing Accuracy` = `% Cases with 100% Testing Accuracy` * 100,
               `Num Incorrect Predictions` = as.numeric(`Num\r\nIncorrect\r\nPredictions`),
               `Run Time (hours)` = RunTime,
               ModelName = factor(paste0("#", Trial, ".", Packages, ".", Layers, "L.", Optimizer, ".", 
                                         Momentum, "m.", LossFunction))) %>%
        select(`Run Date`, `Run Time (hours)`, Trial, Packages, Optimizer, Momentum, LossFunction, Layers,
               LearnRate, `% Cases with\n100% Testing Accuracy`, TestingAccuracy,
               `Num Incorrect Predictions`, ModelName) %>%
        separate(`Run Time (hours)`, into = paste("RT", 1:2, sep = "h")) %>%
        mutate(RTh1 = as.numeric(str_replace(RTh1, "h", "")),
               RTh2 = round(as.numeric(str_replace(RTh2, "m", "")) / 60, 2),
               `Run Time (hours)` = RTh1 + RTh2) %>%
        select(-RTh1, -RTh2)
}

data <- read_data(SELECTED_TRIALS, DOT_IN_FILENAME)
  
# print(data)


# ---------------------------------------------------------------------------------------------
# Dot Plot of Model Performance vs Run Time
# ---------------------------------------------------------------------------------------------
model_perf <- data %>%
    select(ModelName, `Num Incorrect Predictions`, `% Cases with\n100% Testing Accuracy`,
           `Run Time (hours)`, Packages, Trial, `Run Date`)

top5models <- c(58, 52, 50, 57, 53)
model_perf$ModelName <- reorder(model_perf$ModelName, -model_perf$`Num Incorrect Predictions`)
model_perf$Percept_grp <- if_else(model_perf$Trial %in% top5models, "G1", "G2")
top5data <- filter(model_perf, Percept_grp == "G1") %>% droplevels()
bot5data <- filter(model_perf, Percept_grp == "G2") %>% droplevels()


# Top left (num incorrect predictions)
gg_dot_top_left <- ggplot(aes(x = `Num Incorrect Predictions`, y = ModelName, color = Packages), 
                          data = top5data) +
    geom_point(size = 6) +
    scale_color_manual(values = VIZ_COLORS) +
    scale_x_continuous(limits = c(8, 22)) +
    labs(x = '', y = '', title = "# Incorrect Predictions") +
    clean_and_simple_theme +
    theme(axis.text.x = element_blank(), 
          plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")

# print(gg_dot_top_left)

# Bot left (num incorrect predictions)
gg_dot_bot_left <- ggplot(aes(x = `Num Incorrect Predictions`, y = ModelName, color = Packages), 
                          data = bot5data) +
    geom_point(size = 6) +
    scale_color_manual(values = VIZ_COLORS) +
    scale_x_continuous(limits = c(8, 22)) +
    labs(x = '', y = '') +
    clean_and_simple_theme +
    theme(legend.position = "none",
          plot.margin = unit(c(0, 0, 0, 0.19), "cm"))

# print(gg_dot_bot_left)


# Top middle (% with 100% testing accuracy)
gg_dot_top_mid <- ggplot(aes(x = `% Cases with\n100% Testing Accuracy`, y = ModelName, color = Packages), 
                          data = top5data) +
    geom_point(size = 6) +
    scale_color_manual(values = VIZ_COLORS) +
    scale_x_continuous(limits = c(0, 70)) +
    labs(x = '', y = '', title = "% Cases with 100% Testing Accuracy") +
    clean_and_simple_theme +
    theme(axis.text = element_blank(), 
          plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")

# print(gg_dot_top_mid)

# Bot middle (% with 100% testing accuracy)
gg_dot_bot_mid <- ggplot(aes(x = `% Cases with\n100% Testing Accuracy`, y = ModelName, color = Packages), 
                          data = bot5data) +
    geom_point(size = 6) +
    scale_color_manual(values = VIZ_COLORS) +
    scale_x_continuous(limits = c(0, 70), label = function(x) {paste0(x, "%")}) +
    labs(x = '', y = '') +
    clean_and_simple_theme +
    theme(axis.text.y = element_blank(), 
          plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")

# print(gg_dot_bot_mid)


# Top right (Run time in Hours)
gg_dot_top_right <- ggplot(aes(x = `Run Time (hours)`, y = ModelName, color = Packages), 
                          data = top5data) +
    geom_point(size = 6) +
    scale_color_manual(values = VIZ_COLORS) +
    scale_x_continuous(limits = c(7, 13)) +
    labs(x = '', y = '', title = "Run Time (hours)") +
    clean_and_simple_theme +
    theme(axis.text = element_blank(),
          plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")

# print(gg_dot_top_right)

# Bot right (Run time in Hours)
gg_dot_bot_right <- ggplot(aes(x = `Run Time (hours)`, y = ModelName, color = Packages), 
                          data = bot5data) +
    geom_point(size = 6) +
    scale_color_manual(values = VIZ_COLORS) +
    scale_x_continuous(limits = c(7, 13)) +
    labs(x = '', y = '') +
    clean_and_simple_theme +
    theme(axis.text.y = element_blank(),
          plot.margin = unit(c(0, 0, 0, 0), "cm"),
          legend.position = "none")

# print(gg_dot_bot_right)


top_plots <- plot_grid(gg_dot_top_left, gg_dot_top_mid, gg_dot_top_right, ncol = 3)
# ggsave("../visualization/model_perf_top.png", width = 11, height = 3)


bot_plots <- plot_grid(gg_dot_bot_left, gg_dot_bot_mid, gg_dot_bot_right, ncol = 3)
# ggsave("../visualization/model_perf_bot.png", width = 11, height = 3)



# ---------------------------------------------------------------------------------------------
# Line plot of performance improvement over time
# ---------------------------------------------------------------------------------------------
SELECTED_TRIALS_LINE <- c(30, 32, 33, 35, 36, 37, 38, 44, 45, 46, 47, 50, 51, 52, 53, 55, 56, 57, 58)
PERSONAL_PC_TRIALS <- c(30, 32, 33, 37)

line_data <- read_data(SELECTED_TRIALS_LINE, LINE_IN_FILENAME)

# print(line_data)

# Line Plot of Model Performance vs Run Time over Time
model_perf_time <- line_data %>%
    select(Trial, ModelName, `Num Incorrect Predictions`, `Run Time (hours)`, Packages, `Run Date`)

model_perf_time$personal_pc <- NA
model_perf_time[model_perf_time$Trial %in% PERSONAL_PC_TRIALS, ]$personal_pc <- 1

# print(select(model_perf_time, Trial, ModelName, `Run Date`, personal_pc))

gg_line <- ggplot(aes(x = `Run Date`, y = `Num Incorrect Predictions`, color = Packages),
                  data = model_perf_time) +
    geom_point()

# print(gg_line)


# ---------------------------------------------------------------------------------------------
# Bar plot of Number of Incorrect Predictions & Average Testing Accuracy
# ---------------------------------------------------------------------------------------------
BAR_PLOT_TRIALS <- c(33, 35, 39, 41, 45, 47, 50, 51, 52, 53, 55, 56, 57, 58)

BAR_VIZ_COLORS <- c("PyTorch" = "chocolate1",
                    "TensorFlow" = "deepskyblue1")

bar_data <- read_data(BAR_PLOT_TRIALS, LINE_IN_FILENAME)

# print(bar_data)

model_bar_preds <- bar_data %>%
    select(`Run Date`, Trial, ModelName, `Num Incorrect Predictions`, Packages) %>%
    group_by(Packages) %>%
    summarize(avg_inc_preds = mean(`Num Incorrect Predictions`)) %>%
    mutate(Packages = case_when(
        Packages == 'PyT' ~ 'PyTorch',
        Packages == 'TF' ~ 'TensorFlow',
    ))

# print(head(model_bar_preds))


gg_bar_preds <- ggplot(aes(x = Packages, y = avg_inc_preds, fill = Packages), 
                       data = model_bar_preds) +
    geom_col(width = 0.85) +
    labs(x = '', y = '', fill = '',
         title = 'Average Number of Incorrect Predictions') +
    scale_fill_manual(values = BAR_VIZ_COLORS) +
    scale_y_continuous(limits = c(0, 25)) +
    clean_and_simple_theme +
    theme(panel.grid.major.x = element_blank(),
          legend.position = "none",
          # axis.text.x = element_text(size = 12),
          # legend.position = "bottom",
          axis.text.x = element_blank(),
          axis.text.y = element_text(size = 12),
          legend.text = element_text(size = 12),
          plot.title = element_text(size = 14),
    )

# print(gg_bar_preds)
# ggsave("../visualization/bar_inc_preds.png", width = 4.5, height = 3.5)


# Average Testing Accuracy
model_bar_test_acc <- bar_data %>%
    select(`Run Date`, Trial, ModelName, TestingAccuracy, Packages) %>%
    group_by(Packages) %>%
    summarize(avg_test_acc = round(mean(TestingAccuracy), 5)) %>%
    mutate(Packages = case_when(
                Packages == 'PyT' ~ 'PyTorch',
                Packages == 'TF' ~ 'TensorFlow'),
           avg_test_acc_label = paste0(avg_test_acc * 100, "%")
    )

# print(head(model_bar_preds))

gg_bar_test_acc <- ggplot(aes(x = Packages, y = avg_test_acc, fill = Packages), 
                          data = model_bar_test_acc) +
    geom_col(width = 0.85) +
    labs(x = '', y = '', fill = '',
         title = 'Average Testing Accuracy') +
    geom_text(aes(label = avg_test_acc_label), vjust = 2.5, size = 5) +
    scale_fill_manual(values = BAR_VIZ_COLORS) +
    scale_y_continuous(limits = c(0, 1), label = function(y) {paste0(y * 100, "%")}) +
    clean_and_simple_theme +
    theme(panel.grid.major.x = element_blank(),
          legend.position = "none",
          axis.text.x = element_blank(),
          axis.text.y = element_text(size = 11),
          legend.text = element_text(size = 11),
          plot.title = element_text(size = 13)
    )

# print(gg_bar_test_acc)
# ggsave("../visualization/bar_test_acc.png", width = 4.25, height = 3)


# ---------------------------------------------------------------------------------------------
# Bar plot of Number of Incorrect Predictions & Average Testing Accuracy by Network Layers
# ---------------------------------------------------------------------------------------------
# Incorrect Predictions
layer_bar_inc_preds <- bar_data %>%
    select(`Run Date`, Trial, ModelName, Layers, `Num Incorrect Predictions`, Packages) %>%
    mutate(Layers = factor(Layers, levels = c(5, 4))) %>%
    group_by(Layers) %>%
    summarize(avg_inc_preds = mean(`Num Incorrect Predictions`))

gg_bar_layer_inc <- ggplot(aes(x = avg_inc_preds, y = Layers), 
                           data = layer_bar_inc_preds) +
    geom_col(width = 0.85, fill = GEORGE_MASON_COLOR) +
    labs(x = '', y = '', fill = '',
         title = 'Average Number of Incorrect Predictions') +
    scale_x_continuous(limits = c(0, 25)) +
    clean_and_simple_theme +
    theme(panel.grid.major.y = element_blank(),
          legend.position = "none",
          # axis.text.x = element_text(size = 12),
          # legend.position = "bottom",
          axis.text = element_text(size = 12),
          legend.text = element_text(size = 12),
          plot.title = element_text(size = 14),
    )

# print(gg_bar_layer_inc)
# ggsave("../visualization/layer_bar_inc_preds.png", width = 4, height = 2.75)


# Test Accuracy
layer_bar_test_acc <- bar_data %>%
    select(`Run Date`, Trial, ModelName, TestingAccuracy, Layers) %>%
    mutate(Layers = factor(Layers, levels = c(5, 4))) %>%
    group_by(Layers) %>%
    summarize(avg_test_acc = round(mean(TestingAccuracy), 5)) %>%
    mutate(avg_test_acc_label = paste0(avg_test_acc * 100, "%"))

gg_bar_layer_test_acc <- ggplot(aes(x = avg_test_acc, y = Layers), 
                           data = layer_bar_test_acc) +
    geom_col(width = 0.85, fill = GEORGE_MASON_COLOR) +
    labs(x = '', y = '', fill = '',
         title = 'Average Testing Accuracy') +
    geom_text(aes(label = avg_test_acc_label), hjust = 1.25, size = 4, color = 'white') +
    scale_x_continuous(limits = c(0, 1), label = function(x) {paste0(x * 100, "%")}) +
    clean_and_simple_theme +
    theme(panel.grid.major.y = element_blank(),
          legend.position = "none",
          # axis.text.x = element_text(size = 12),
          # legend.position = "bottom",
          axis.text = element_text(size = 12),
          legend.text = element_text(size = 12),
          plot.title = element_text(size = 14),
    )

# print(gg_bar_layer_test_acc)
ggsave("../visualization/layer_bar_test_acc.png", width = 3.25, height = 2.75)
