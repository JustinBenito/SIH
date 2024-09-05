const Hero = () => {
  return (
    <div className="relative [background:linear-gradient(110.8deg,_#ffc989,_#fffcb9_48.96%,_rgba(255,_255,_255,_0))] w-full h-[781.31px] flex flex-col py-6 px-[76px] box-border items-center justify-center text-left text-41xl text-black font-roboto md:w-auto md:[align-self:unset] md:h-auto md:items-center md:justify-center">
      <div className="self-stretch flex-1 flex flex-col items-start justify-start">
        <div className="self-stretch flex-1 flex flex-row items-center justify-start md:h-auto md:flex-col md:items-center md:justify-end">
          <div className="flex-1 flex flex-col items-start justify-center gap-[37px] md:flex-[unset] md:self-stretch">
            <h3 className="m-0 self-stretch relative text-[inherit] leading-[75.5px] capitalize font-inherit">
              <p className="m-0">
                <span className="font-medium">{`Unlock Worlds 1st `}</span>
                <span className="font-extrabold font-roboto">
                  <span className="text-darkorange-200">AI</span>
                  <span className="text-aquamarine">{` `}</span>
                </span>
              </p>
              <p className="m-0 font-medium">powered Edu-Platform!</p>
            </h3>
            <div className="self-stretch relative text-lg leading-[130%] font-inter text-gray-400">
              Empowering learners worldwide with curated YouTube courses,
              personalized paths, and collaborative community for inclusive,
              accessible, and transformative education.
            </div>
            <div className="self-stretch flex flex-row items-center justify-start gap-[11px]">
              <button className="cursor-pointer [border:none] py-[18px] px-[45px] bg-darkorange-200 rounded-31xl shadow-[0px_10px_50px_rgba(255,_255,_255,_0.25)] flex flex-row items-start justify-start">
                <b className="relative text-lg font-dm-sans text-text-white text-center">
                  Explore
                </b>
              </button>
              <button className="cursor-pointer py-[18px] px-[42px] bg-[transparent] rounded-31xl shadow-[0px_10px_50px_rgba(255,_255,_255,_0.25)] flex flex-row items-start justify-start border-[1.5px] border-solid border-darkorange-200">
                <b className="relative text-lg font-dm-sans text-darkorange-200 text-center">
                  Learn More
                </b>
              </button>
            </div>
          </div>
          <div className="self-stretch flex-1 flex flex-col p-2.5 items-center justify-center md:flex-[unset] md:self-stretch">
            <div className="relative w-[640.39px] h-[713.31px]">
              <div className="absolute top-[106.14px] left-[119.27px] rounded-[248px] [background:linear-gradient(94.64deg,_#fae3c7,_#fffed7)] shadow-[4px_4px_4px_#fff] [backdrop-filter:blur(4px)] box-border w-[437px] h-[407px] border-[1px] border-solid border-text-white" />
              <img
                className="absolute top-[0px] left-[0px] w-[507px] h-[494px] object-cover"
                alt=""
                src="/aka@2x.png"
              />
              <img
                className="absolute top-[267px] left-[105.76px] w-[124.43px] h-[125.18px]"
                alt=""
                src="/yellow-blob.svg"
              />
              <img
                className="absolute top-[192.04px] left-[119.11px] w-[521.28px] h-[521.27px] object-cover"
                alt=""
                src="/rocket@2x.png"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
