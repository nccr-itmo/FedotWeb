import React, { useEffect } from "react";
import { useWindowSize } from "use-hooks";
import { useHistory } from "react-router-dom";
import { useSelector } from "react-redux";
import scss from "./signInPage.module.scss";

import SignInPageTitle from "./title/SignInPageTitle";
import SignInPagePaper from "./paper/SignInPagePaper";
import { StateType } from "../../../redux/store";

const SignInPage = () => {
  const history = useHistory();
  const size = useWindowSize();

  const { user_info } = useSelector((state: StateType) => state.auth);

  useEffect(() => {
    if (user_info !== null) {
      history.push("/ws");
    }
  }, [history, user_info]);
  return (
    <section className={scss.root} style={{ minHeight: size.height }}>
      <SignInPageTitle />
      <SignInPagePaper />
    </section>
  );
};

export default SignInPage;
