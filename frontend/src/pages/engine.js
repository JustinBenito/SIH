import React, { useEffect } from 'react';

const SphereEngineWidget = () => {
  useEffect(() => {
    SE_BASE = "widgets.sphere-engine.com";
    SE_HTTPS = true;
    SE = window.SE || (window.SE = []);
    var js, fjs = document.getElementsByTagName("script")[0];
    if (document.getElementById("sphere-engine-jssdk")) return;
    js = document.createElement("script"); js.id = "sphere-engine-jssdk";
    js.src = (SE_HTTPS ? "https" : "http") + "://" + SE_BASE + "/static/sdk/sdk.min.js";
    fjs.parentNode.insertBefore(js, fjs);
  }, []);

  return null;
};

export default SphereEngineWidget;
