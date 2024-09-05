import { useCallback } from "react";
import { useNavigate } from "react-router-dom";

const Lang = () => {
  const navigate = useNavigate();

  const onButtonStyle1FilledRightIcClick = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  const onButtonStyle1FilledRightIc2Click = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  const onButtonStyle1FilledRightIc3Click = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  const onButtonStyle1FilledRightIc4Click = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  const onButtonStyle1FilledRightIc5Click = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  const onButtonStyle1FilledRightIc6Click = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  const onButtonStyle1FilledRightIc7Click = useCallback(() => {
    navigate("/login");
  }, [navigate]);

  return (
    <div className="relative [background:linear-gradient(106.37deg,_#ffc989,_#fffcb9_48.96%,_rgba(255,_255,_255,_0))] w-full h-[852px] flex flex-col py-[75px] px-36 box-border items-center justify-center gap-[30px] text-center text-41xl text-black font-roboto">
      <div className="self-stretch flex flex-col items-center justify-center">
        <div className="self-stretch relative leading-[75.5px] capitalize font-medium">
          Feel like Learning?
        </div>
      </div>
      <div className="self-stretch flex flex-col py-2.5 px-0 items-center justify-center gap-[70px]">
        <div className="self-stretch flex flex-col items-center justify-center">
          <div className="self-stretch flex flex-row items-center justify-between">
            <div className="rounded-xl bg-text-white flex flex-row items-center justify-center">
              <div className="rounded-mini w-[243px] h-[257px] flex flex-col items-center justify-center gap-[32px]">
                <img
                  className="relative w-[125px] h-[127px] overflow-hidden shrink-0"
                  alt=""
                  src="/python.svg"
                />
                <button
                  className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center"
                  onClick={onButtonStyle1FilledRightIcClick}
                >
                  <div className="flex flex-row items-end justify-start gap-[6px]">
                    <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">
                      Python
                    </div>
                    <img
                      className="relative w-3.5 h-3.5 hidden"
                      alt=""
                      src="/arrowbackwardcircle3.svg"
                    />
                  </div>
                </button>
              </div>
            </div>
            <div className="rounded-xl bg-text-white flex flex-row items-center justify-center">
              <div className="rounded-mini w-[244px] h-64 flex flex-col items-center justify-center gap-[32px]">
                <img
                  className="relative w-[125px] h-[127px] overflow-hidden shrink-0"
                  alt=""
                  src="/angular.svg"
                />
                <button className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center">
                  <div className="flex flex-row items-end justify-start gap-[6px]">
                    <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">
                      Angular
                    </div>
                    <img
                      className="relative w-3.5 h-3.5 hidden"
                      alt=""
                      src="/arrowbackwardcircle3.svg"
                    />
                  </div>
                </button>
              </div>
            </div>
            <div className="rounded-xl bg-text-white w-60 h-64 flex flex-col items-center justify-center gap-[32px]">
              <img
                className="relative w-[125px] h-[124.34px] object-cover"
                alt=""
                src="/programming-logos-and-icons-by-hrhasnai-5@2x.png"
              />
              <button
                className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center"
                onClick={onButtonStyle1FilledRightIc2Click}
              >
                <div className="flex flex-row items-end justify-start gap-[6px]">
                  <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">
                    HTML 5
                  </div>
                  <img
                    className="relative w-3.5 h-3.5 hidden"
                    alt=""
                    src="/arrowbackwardcircle3.svg"
                  />
                </div>
              </button>
            </div>
            <div className="rounded-xl bg-text-white w-[234px] h-64 flex flex-col items-center justify-center gap-[32px]">
              <img
                className="relative w-[125px] h-[125px] object-cover"
                alt=""
                src="/programming-logos-and-icons-by-hrhasnai-11@2x.png"
              />
              <button
                className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center"
                onClick={onButtonStyle1FilledRightIc3Click}
              >
                <div className="flex flex-row items-end justify-start gap-[6px]">
                  <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">{`JavaScript `}</div>
                  <img
                    className="relative w-3.5 h-3.5 hidden"
                    alt=""
                    src="/arrowbackwardcircle3.svg"
                  />
                </div>
              </button>
            </div>
          </div>
        </div>
        <div className="self-stretch flex flex-row items-center justify-between">
          <div className="flex-1 h-[263px] flex flex-row items-center justify-between">
            <div className="rounded-xl bg-text-white w-[244px] h-[267px] flex flex-col items-center justify-center gap-[32px]">
              <img
                className="relative w-[125px] h-[127px] overflow-hidden shrink-0"
                alt=""
                src="/24.svg"
              />
              <button
                className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center"
                onClick={onButtonStyle1FilledRightIc4Click}
              >
                <div className="flex flex-row items-end justify-start gap-[6px]">
                  <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">
                    Java
                  </div>
                  <img
                    className="relative w-3.5 h-3.5 hidden"
                    alt=""
                    src="/arrowbackwardcircle3.svg"
                  />
                </div>
              </button>
            </div>
            <div className="rounded-xl bg-text-white w-[241px] h-[263px] flex flex-col items-center justify-center gap-[32px]">
              <img
                className="relative w-[125px] h-[127px] overflow-hidden shrink-0"
                alt=""
                src="/35.svg"
              />
              <button
                className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center"
                onClick={onButtonStyle1FilledRightIc5Click}
              >
                <div className="flex flex-row items-end justify-start gap-[6px]">
                  <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">
                    React
                  </div>
                  <img
                    className="relative w-3.5 h-3.5 hidden"
                    alt=""
                    src="/arrowbackwardcircle3.svg"
                  />
                </div>
              </button>
            </div>
            <div className="rounded-xl bg-text-white w-[249px] h-[257px] flex flex-col items-center justify-center gap-[32px]">
              <img
                className="relative w-[125px] h-[127px] overflow-hidden shrink-0"
                alt=""
                src="/1.svg"
              />
              <button
                className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center"
                onClick={onButtonStyle1FilledRightIc6Click}
              >
                <div className="flex flex-row items-end justify-start gap-[6px]">
                  <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">
                    C
                  </div>
                  <img
                    className="relative w-3.5 h-3.5 hidden"
                    alt=""
                    src="/arrowbackwardcircle4.svg"
                  />
                </div>
              </button>
            </div>
            <div className="rounded-xl bg-text-white w-[244px] h-[255px] flex flex-col items-center justify-center gap-[32px]">
              <img
                className="relative w-[125px] h-[127px] overflow-hidden shrink-0"
                alt=""
                src="/3.svg"
              />
              <button
                className="cursor-pointer [border:none] py-5 px-[52px] bg-darkorange-200 rounded-2xl flex flex-row items-center justify-center"
                onClick={onButtonStyle1FilledRightIc7Click}
              >
                <div className="flex flex-row items-end justify-start gap-[6px]">
                  <div className="relative text-xl leading-[16px] font-semibold font-caption-heavy text-text-white text-left">
                    C++
                  </div>
                  <img
                    className="relative w-3.5 h-3.5 hidden"
                    alt=""
                    src="/arrowbackwardcircle5.svg"
                  />
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Lang;
