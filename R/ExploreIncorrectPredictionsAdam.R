# Explore Incorrect Predictions from 2 PyTorch models:
#
# Adam (49 incorrect)
#     (3-512-128-64-100)  [4 layers]
#     Relu activation
#     Adam optimizer
#     2250 epochs
#     80/20 training/testing split


library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
source("clean_ggplot.R")


GEORGE_MASON_COLOR <- "#116020"


data <- read_excel("../model_output/PyTorch_800caseResults__10-05-2022_223013.xlsx", sheet="RawResults") %>%
    filter(CorrectPred == FALSE) %>%
    select(CaseNum, xTemp, yVol, direction, Target, Prediction, CorrectPred) %>%
    mutate(xTemp = factor(xTemp),
           yVol = factor(yVol, levels = c(9, 8, 7, 6, 5, 4, 3, 2, 1, 0)),
           direction = direction)

# print(head(data))


# Bar plot of # Cases predicted Incorrectly by Initial X, Y, Direction
gg_x <- ggplot(data, aes(x=xTemp)) + 
    geom_bar(fill = GEORGE_MASON_COLOR) + 
    labs(x = '', y = '# of Cases',
         title = '# Incorrect Predictions by Initial Condition (X - Temperature)') +
    scale_y_continuous(breaks = c(0, 2, 4, 6, 8, 10), limits = c(0, 10)) +
    clean_and_simple_theme +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          plot.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          axis.title = element_text(size = 14))

# print(gg_x)
# ggsave("../eda/PyT_Adam_wrong_preds_X.png")


# Y - Volume
gg_y <- ggplot(data, aes(x=yVol)) + 
    geom_bar(fill = GEORGE_MASON_COLOR) + 
    labs(x = '', y = '# of Cases',
         title = '# Incorrect Predictions by Initial Condition (Y - Volume)') +
    # scale_y_continuous(breaks = c(0, 2, 4, 6, 8), limits = c(0, 8)) +
    clean_and_simple_theme +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          plot.title = element_text(size = 16),
          axis.text = element_text(size = 14),
          axis.title = element_text(size = 14)) +
    coord_flip() 

# print(gg_y)
# ggsave("../eda/PyT_Adam_wrong_preds_Y.png")


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

df_point <- data.frame(x = c(10, 10, -10, -10), y = c(-10, 10, -10, 10))

gg_dir <- ggplot(direction_data) +
    geom_point(aes(x=x, y=y), color = 'white', size = 0.1, df_point) +
    # scale_x_continuous(breaks = c(-4, -2, 0, 2, 4)) +
    # scale_y_continuous(breaks = c(-4, -2, 0, 2, 4)) +
    
    # North
    geom_segment(aes(x = 0, y = 0, xend = 0, yend = 7),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 0, y = 7.75, label = "North (7)", color = "black") +
    
    # North-East
    geom_segment(aes(x = 0, y = 0, xend = 4 / sqrt(2), yend = 4 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 5, y = 3, label = "North-East (4)", color = "black") +
    
    # East
    geom_segment(aes(x = 0, y = 0, xend = 6, yend = 0),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 7.5, y = 0, label = "East (6)", color = "black") +

    # South-East
    geom_segment(aes(x = 0, y = 0, xend = 7 / sqrt(2), yend = - 7 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 6.65, y = -5.5, label = "South-East (7)", color = "black") +

    # South
    geom_segment(aes(x = 0, y = 0, xend = 0, yend = - 3),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = 0, y = -3.75, label = "South (3)", color = "black") +

    # South-West
    geom_segment(aes(x = 0, y = 0, xend = - 6 / sqrt(2), yend = - 6 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = -5.75, y = -5, label = "South-West (6)", color = "black") +

    # West
    geom_segment(aes(x = 0, y = 0, xend = - 5, yend = 0),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = -6.5, y = 0, label = "West (5)", color = "black") +

    # North-West
    geom_segment(aes(x = 0, y = 0, xend = - 11 / sqrt(2), yend = 11 / sqrt(2)),
        arrow = arrow(length = unit(0.5, "cm")), size = 2, color = GEORGE_MASON_COLOR) +
    annotate(geom = "text", x = -8, y = 8.5, label = "North-West (11)", color = "black") +

    labs(x = '', y = '', title = '# Incorrect Predictions by Initial Condition (Direction)') +
    clean_and_simple_theme +
    theme(plot.title = element_text(hjust = 0.5, size = 16),
          axis.ticks = element_blank(),
          axis.text = element_blank(),
          axis.title = element_blank())


# print(gg_dir)
# ggsave("../eda/PyT_Adam_wrong_preds_Dir.png")