library(tidyverse)
library(lubridate)

# Read your CSV
df <- read.csv("Coonamessett.csv")
df <- read.csv("Ipswich.csv")
df <- read.csv("Santuit.csv")

# ---- 1. Clean datetime ----
# Extract date & time from 'name'
df <- df %>%
  mutate(
    # extract "2024-05-07_05-14-34"
    dt_string = str_extract(name, "\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2}"),
    # convert to proper datetime
    datetime = ymd_hms(str_replace(dt_string, "_", " ")),
    date = as.Date(datetime),
    date_within_year = as.Date(paste0("2000-", month(date), "-", day(date))),
    year = year(date) 
  )


# Bar plot: number of videos per time_of_day, separated by train/val/test
# ---- 2. Plot: Count per time of day ----
p1 <- df %>%
  count(time_of_day, train_set) %>%
  ggplot(aes(x = time_of_day, y = n, fill = train_set)) +
  geom_col(position = "dodge") +
  labs(
    x = "Time of Day",
    y = "Number of Videos",
    fill = "Dataset Split",
    title = "Time of Day"
  ) +
  theme_minimal(base_size = 14)

p1


# Seasonal distribution (histogram over dates)
p2 <- df %>%
  ggplot(aes(x = date_within_year, fill = train_set)) +
  geom_histogram(binwidth = 1, position = "stack", color = "black") +
  #facet_wrap(~ year) +
  labs(
    x = "Date",
    y = "Number of Videos",
    fill = "Dataset Split",
    title = "Seasonal Distribution"
  ) +
  theme_minimal(base_size = 14)

p2

library(gridExtra)
p1 <- p1 + theme(legend.position = "none")
p3 <- grid.arrange(p1, p2, ncol=2,
                   widths = c(2, 3))

# ---------------------------------------------------------
# Save the plot to a PNG file
ggsave(
  filename = "Santuit.jpg",  # File name
  plot = p3,                  # The plot object
  width = 8,                 # Width in inches
  height = 4,                # Height in inches
  dpi = 300,                 # Resolution in dots per inch
  units = "in"               # Units for width and height
)
