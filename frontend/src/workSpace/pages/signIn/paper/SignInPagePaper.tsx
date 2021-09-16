import React from "react";
import scss from "./signInPagePaper.module.scss";
import { useDispatch } from "react-redux";

import { Paper } from "@material-ui/core";
import SignInForm from "../../../components/forms/signIn/SignInForm";
import { authUser } from "../../../../redux/reducers/auth/authReducer";

const SignInPagePaper = () => {
  const dispatch = useDispatch();

  const signInGuest = () => {
    dispatch(authUser("guest", "guest"));
  };

  return (
    <section>
      <Paper elevation={3} className={scss.root}>
        <p className={scss.title}>Sign In</p>
        <SignInForm />
        <div className={scss.asGuestPosition}>
          <p className={scss.asGuest} onClick={signInGuest}>
            Sign In as guest
          </p>
        </div>
      </Paper>
    </section>
  );
};

export default SignInPagePaper;
