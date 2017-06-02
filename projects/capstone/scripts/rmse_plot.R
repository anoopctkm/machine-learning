library(ggplot2)

d <- data.frame(
  Model = c("Benchmark", "Content-based", "Hybrid"),
  RMSE  = c(.9396, .8712, .9784)
)

ggsave("rmse_plot.png", width = 6, height = 5, plot = 
   ggplot(d, aes(Model, RMSE, label = corrr::fashion(RMSE))) +
   geom_col(show.legend = FALSE, fill = "#0091ff", color = "black") +
   geom_text(color = "white", vjust = 2, size = 6) +
   theme_minimal() +
   coord_cartesian(ylim = c(.8, 1)) +
   ggtitle("Comparison of model performance.",
           subtitle = "Lower RMSE is better")
)
