import React, { memo } from "react";
import ReactApexChart from "react-apexcharts";

const SandboxPageChartsMetricLine = ({ data }) => {
  const series = data.series;

  const options = {
    chart: {
      height: 350,
      type: "line",
      zoom: {
        enabled: false,
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      curve: "straight",
    },
    grid: {
      row: {
        colors: ["#f3f3f3", "transparent"], // takes an array which will be repeated on columns
        opacity: 0.5,
      },
    },
    xaxis: data.options.xaxis,
    yaxis: data.options.yaxis,
  };

  return (
    <div id="chart1">
      <ReactApexChart
        options={options}
        series={series}
        type="line"
        height={350}
      />
    </div>
  );
};

export default memo(SandboxPageChartsMetricLine);
