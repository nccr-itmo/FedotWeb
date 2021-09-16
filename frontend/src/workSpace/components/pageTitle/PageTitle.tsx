import React, { FC, memo } from "react";
import scss from "./pageTitle.module.scss";

interface IPageTitle {
  title: string;
}

const PageTitle: FC<IPageTitle> = ({ title }) => {
  return (
    <div className={scss.pageTitle}>
      <p className={scss.title}>{title}</p>
      <div className={scss.line} />
    </div>
  );
};

export default memo(PageTitle);
