import React, { memo, useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";

import scss from "./historyPageChart.module.scss";

import HistoryPageChartBoxplot from "./boxplot/HistoryPageChartBoxplot";
import HistoryPageChartHeader from "./header/HistoryPageChartHeader";
import { StateType } from "../../../../redux/store";
import {
  getGenerations,
  setGenerations,
} from "../../../../redux/reducers/history/historyReducer";

export const editArr = (arr: any) => {
  return arr.map((item: any) => {
    item.name = `${item.name} epoch`;
    return item;
  });
};

const HistoryPageChart = () => {
  const dispatch = useDispatch();
  const [pheno, setPheno] = useState();
  const [geno, setGeno] = useState();

  const {
    history_genothypic_phenothypic,
    history_generations_pheno,
    history_generations_geno,
  } = useSelector((state: StateType) => state.history);

  const { show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );
  useEffect(() => {
    dispatch(setGenerations(null, "pheno"));
    dispatch(setGenerations(null, "geno"));

    if (show_case_by_id) {
      dispatch(getGenerations(show_case_by_id.case_id, "pheno"));
      dispatch(getGenerations(show_case_by_id.case_id, "geno"));
    }
  }, [dispatch, show_case_by_id]);

  useEffect(() => {
    if (history_generations_pheno && history_generations_geno) {
      setPheno(editArr(history_generations_pheno.series));
      setGeno(editArr(history_generations_geno.series));
    }
  }, [history_generations_pheno, history_generations_geno]);

  return (
    <div className={scss.root}>
      <HistoryPageChartHeader />
      <HistoryPageChartBoxplot
        data={history_genothypic_phenothypic ? pheno : geno}
      />
    </div>
  );
};

export default memo(HistoryPageChart);
