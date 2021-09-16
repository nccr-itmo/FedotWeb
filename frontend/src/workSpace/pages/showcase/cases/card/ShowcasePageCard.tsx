import React, { FC, memo } from "react";
import scss from "./showcasePageCard.module.scss";

import CardButton from "../../../../../ui/buttons/CardButton";
import { ICaseArr } from "../../../../../API/showCase/showCasesInterface";

interface IShowcasePageCard {
  caseItem: ICaseArr;
  onClick(value: string): void;
}

const ShowcasePageCard: FC<IShowcasePageCard> = ({ caseItem, onClick }) => {
  return (
    <div className={scss.card}>
      <img src={caseItem.icon_path} alt={caseItem.title} className={scss.img} />
      <div className={scss.titleAndButton}>
        <p className={scss.name}>{caseItem.title}</p>
        <CardButton id={caseItem.case_id} onClick={onClick}>
          Expand
        </CardButton>
      </div>
    </div>
  );
};

export default memo(ShowcasePageCard);
