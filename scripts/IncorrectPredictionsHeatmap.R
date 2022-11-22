# Date Created: Nov 18, 2022
#
# Use NN Model Tracker Excel file to make a heat map of starting X, Y, and direction for
# incorrectly predicted cases from top 10 best performing models
# 

library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(RColorBrewer)
source("clean_ggplot.R")


INPUT_PATH <- "../heatmap_data/"

VIZ_COLORS <- c("Py" = "chocolate1",
                "TF" = "deepskyblue1")

clean_heatmap_theme <- theme_gray() + theme(
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(hjust = -.5),

    strip.background = element_rect(fill = rgb(.9,.95,1),
        colour = gray(.5), size = .2),

    panel.border = element_rect(fill = FALSE,colour = gray(.70)),
    panel.spacing.x = unit(0.10,"cm"),
    panel.spacing.y = unit(0.05,"cm"),

    axis.ticks=element_blank()
)


# Read in all 10 Excel files
xl_files <- list.files(path = INPUT_PATH, all.files = FALSE, full.names = TRUE)

load_xl_data <- function(fileNumber) {
    data <- read_excel(xl_files[fileNumber], sheet = 'RawResults')
    filename <- unlist(strsplit(xl_files[fileNumber], "/"))[[3]]
    trial <- unlist(strsplit(filename, "_"))[[1]]
    package <- unlist(strsplit(filename, "_"))[[2]]

    stamped <- data %>%
        mutate(trial = trial,
               package = package,
               xTemp = factor(xTemp),
               yVol = factor(yVol),
               dir_X = case_when(
                    direction == 0 ~ 1.5,
                    direction == 1 ~ 2.5,
                    direction == 2 ~ 2.5,
                    direction == 3 ~ 2.5,
                    direction == 4 ~ 1.5,
                    direction == 5 ~ 0.5,
                    direction == 6 ~ 0.5,
                    direction == 7 ~ 0.5),

               dir_Y = case_when(
                    direction == 0 ~ 2.5,
                    direction == 1 ~ 2.5,
                    direction == 2 ~ 1.5,
                    direction == 3 ~ 0.5,
                    direction == 4 ~ 0.5,
                    direction == 5 ~ 0.5,
                    direction == 6 ~ 1.5,
                    direction == 7 ~ 2.5)) %>%
        filter(CorrectPred == FALSE) %>%
        select(xTemp, yVol, direction, dir_X, dir_Y)
}

all_data = load_xl_data(1)

for (num in 2:10) {
    xl_data = load_xl_data(num)
    all_data = rbind(all_data, xl_data)
}

heat_data <- all_data %>%
    group_by(xTemp, yVol, direction, dir_X, dir_Y) %>%
    count()

# print(heat_data)

# ---------------------------------------------------------------------------------------------
# Heatmap of Incorrect Predictions by Initial Conditions
# ---------------------------------------------------------------------------------------------
gg_heat <- ggplot(aes(x = dir_X, y = dir_Y, color = n), data = heat_data) +
    geom_point(shape = 15, size = 8) +
    labs(x = 'X (Temperature)', y = 'Y (Volume)') +
    facet_grid(yVol ~ xTemp, switch = 'both') +
    scale_x_continuous(limits = c(0, 3), breaks = c(0, 1, 2, 3)) +
    scale_y_continuous(limits = c(0, 3), breaks = c(0, 1, 2, 3)) +
    # scale_color_gradient2(low = 'blue', mid = '#FFE699', high = 'red') +
    scale_color_gradient(low = 'lemonchiffon', high = 'darkred') +
    clean_heatmap_theme +
    theme(panel.grid.minor = element_blank(),
          legend.position = "none",
          axis.text = element_blank(),
          axis.title = element_text(size = 14),
          strip.text = element_text(size = 14)) 

# print(gg_heat)

ggsave("../visualization/incorrectPredsHeatmapGray.png")


arrange(heat_data, desc(n))