# Explore Incorrect Predictions from a PyTorch models:
# 1) Best performer (19 incorrect)
#     (3-512-128-64-100)  [4 layers]
#     Relu activation
#     SGD optimizer
#     2250 epochs
#     80/20 training/testing split


library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
source("clean_ggplot.R")


GEORGE_MASON_COLOR <- "#116020"


data <- read_excel("../model_output/PyTorch_800caseResultsBounce__09-19-2022_163431.xlsx", sheet="RawResults") %>%
    filter(CorrectPred == FALSE) %>%
    select(CaseNum, xTemp...2, yVol...3, direction...4, Target, Prediction, CorrectPred) %>%
    mutate(xTemp = factor(xTemp...2),
           yVol = factor(yVol...3, levels = c(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)),
           direction = direction...4) %>%
    select(-xTemp...2, -yVol...3, -direction...4)

# print(head(data))


# Bar plot of # Cases predicted Incorrectly by Initial X, Y, Direction
gg_x <- ggplot(data, aes(x=xTemp)) + 
    geom_bar(fill = GEORGE_MASON_COLOR) + 
    labs(x = '', y = '# of Cases',
         title = '# Incorrect Predictions by Initial Condition (X - Temperature)') +
    scale_y_continuous(breaks = c(0, 2, 4, 6, 8, 10, 12, 14), limits = c(0, 14)) +
    clean_and_simple_theme +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          plot.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          axis.title = element_text(size = 14))

# print(gg_x)
# ggsave("../eda/PyT_wrong_preds_X.png")


# Y - Volume
gg_y <- ggplot(data, aes(x=yVol)) + 
    geom_bar(fill = GEORGE_MASON_COLOR) + 
    labs(x = '', y = '# of Cases',
         title = '# Incorrect Predictions by Initial Condition (Y - Volume)') +
    scale_y_continuous(breaks = c(0, 2, 4, 6, 8), limits = c(0, 8)) +
    clean_and_simple_theme +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          plot.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          axis.title = element_text(size = 14)) +
    coord_flip() 

# print(gg_y)
# ggsave("../eda/PyT_wrong_preds_Y.png")


# Direction init
direction_data <- data %>%
    mutate(direction_text = case_when(
        direction == 0 ~ "North",
        direction == 1 ~ "North-East",
        direction == 2 ~ "East",
        direction == 3 ~ "South-East",
        direction == 4 ~ "South",
        direction == 5 ~ "South-West",
        direction == 6 ~ "West",
        direction == 7 ~ "North-West"
    )) %>%
    group_by(direction, direction_text) %>%
    count()

df_point <- data.frame(x = c(4.25, 4.25, -4.25, -4.25), y = c(-4.25, 4.25, -4.25, 4.25))

gg_dir <- ggplot(direction_data) +
    geom_point(aes(x=x, y=y), color = 'white', size = 0.1, df_point) +
    scale_x_continuous(breaks = c(-4, -2, 0, 2, 4)) +
    scale_y_continuous(breaks = c(-4, -2, 0, 2, 4)) +
    
    # North
    geom_segment(aes(x = 0, y = 0, xend = 0, yend = 4),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 0, y = 4.25, label = "North (4)", color = "black") +
    
    # North-East
    geom_segment(aes(x = 0, y = 0, xend = 1 / sqrt(2), yend = 1 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 1.5, y = 1, label = "North-East (1)", color = "black") +
    
    # East
    geom_segment(aes(x = 0, y = 0, xend = 1, yend = 0),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 1.6, y = 0, label = "East (1)", color = "black") +

    # South-East
    geom_segment(aes(x = 0, y = 0, xend = 2 / sqrt(2), yend = - 2 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 2.25, y = -1.5, label = "South-East (2)", color = "black") +

    # South
    geom_segment(aes(x = 0, y = 0, xend = 0, yend = - 3),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 0, y = -3.25, label = "South (3)", color = "black") +

    # South-West
    geom_segment(aes(x = 0, y = 0, xend = - 4 / sqrt(2), yend = - 4 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = -3.65, y = -3, label = "South-West (4)", color = "black") +

    # West
    geom_segment(aes(x = 0, y = 0, xend = - 1, yend = 0),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = -1.6, y = 0, label = "West (1)", color = "black") +

    # North-West
    geom_segment(aes(x = 0, y = 0, xend = - 3 / sqrt(2), yend = 3 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = -3, y = 2.25, label = "North-West (3)", color = "black") +

    labs(x = '', y = '', title = '# Incorrect Predictions by Initial Condition (Direction)') +
    clean_and_simple_theme +
    theme(plot.title = element_text(hjust = 0.5, size = 16),
          axis.ticks = element_blank(),
          axis.text = element_blank(),
          axis.title = element_blank())


# print(gg_dir)
# ggsave("../eda/PyT_wrong_preds_Dir.png")