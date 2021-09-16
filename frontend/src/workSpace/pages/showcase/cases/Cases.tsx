import React, { memo, useCallback, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import clsx from "clsx";

import scss from "./cases.module.scss";

import PageTitle from "../../../components/pageTitle/PageTitle";
import ShowcasePageCard from "./card/ShowcasePageCard";
import { StateType } from "../../../../redux/store";

import { setHistoryToggle } from "./../../../../redux/reducers/sandBox/sandBoxReducer";
import {
  getShowCaseById,
  getShowCasesArr,
} from "../../../../redux/reducers/showCase/showCaseReducer";

const Cases = () => {
  const dispatch = useDispatch();
  const { show_cases_arr, show_case_by_id } = useSelector(
    (state: StateType) => state.showCases
  );

  useEffect(() => {
    dispatch(getShowCasesArr());
  }, [dispatch]);

  const handleSelectCase = useCallback(
    (caseId: string): void => {
      dispatch(getShowCaseById(caseId));
      dispatch(setHistoryToggle(false));
    },
    [dispatch]
  );

  return (
    <section
      className={
        show_case_by_id ? clsx(scss.showcase, scss.selected) : scss.showcase
      }
    >
      <PageTitle title="Case" />
      {show_cases_arr ? (
        <div className={scss.cardsArea}>
          {show_cases_arr.map((caseItem, index) => (
            <div key={index + "l"} className={scss.cardsAreaItem}>
              <ShowcasePageCard
                caseItem={caseItem}
                onClick={handleSelectCase}
              />
            </div>
          ))}
        </div>
      ) : (
        <></>
      )}
    </section>
  );
};

export default memo(Cases);
